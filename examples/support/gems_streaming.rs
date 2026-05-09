use std::{
    collections::{BTreeMap, VecDeque},
    env, fmt,
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HostWindowPlan {
    pub(crate) sequence: usize,
    pub(crate) start_item: usize,
    pub(crate) items: usize,
}

pub(crate) fn host_window_plan(total_items: usize, window_items: usize) -> Vec<HostWindowPlan> {
    if total_items == 0 {
        return Vec::new();
    }
    let window_items = window_items.max(1);
    let mut plans = Vec::with_capacity(total_items.div_ceil(window_items));
    let mut start_item = 0usize;
    while start_item < total_items {
        let items = window_items.min(total_items - start_item);
        plans.push(HostWindowPlan {
            sequence: plans.len(),
            start_item,
            items,
        });
        start_item += items;
    }
    plans
}

#[derive(Debug, Clone)]
pub(crate) struct LoaderWorkerError {
    message: String,
}

impl LoaderWorkerError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub(crate) fn with_worker(self, worker_id: usize) -> Self {
        Self {
            message: format!("worker {worker_id}: {}", self.message),
        }
    }
}

impl fmt::Display for LoaderWorkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for LoaderWorkerError {}

impl From<std::io::Error> for LoaderWorkerError {
    fn from(error: std::io::Error) -> Self {
        Self::new(error.to_string())
    }
}

struct HostWindowMessage<W> {
    sequence: usize,
    result: Result<W, LoaderWorkerError>,
}

pub(crate) struct OrderedHostWindowStream<W> {
    receiver: Option<Receiver<HostWindowMessage<W>>>,
    buffered: BTreeMap<usize, W>,
    next_sequence: usize,
    total_windows: usize,
    joins: Vec<JoinHandle<()>>,
}

impl<W> OrderedHostWindowStream<W> {
    pub(crate) fn next_window(&mut self, kind: &str) -> Option<W> {
        if self.next_sequence >= self.total_windows {
            return None;
        }
        if let Some(window) = self.buffered.remove(&self.next_sequence) {
            self.next_sequence += 1;
            return Some(window);
        }

        loop {
            let receiver = self
                .receiver
                .as_ref()
                .expect("host-window receiver should be present while streaming");
            let message = receiver.recv().unwrap_or_else(|error| {
                panic!("{kind} host workers stopped before producing all windows: {error}")
            });
            match message.result {
                Ok(window) if message.sequence == self.next_sequence => {
                    self.next_sequence += 1;
                    return Some(window);
                }
                Ok(window) => {
                    self.buffered.insert(message.sequence, window);
                    if let Some(window) = self.buffered.remove(&self.next_sequence) {
                        self.next_sequence += 1;
                        return Some(window);
                    }
                }
                Err(error) => {
                    panic!("{kind} host worker failed: {error}");
                }
            }
        }
    }
}

impl<W> Drop for OrderedHostWindowStream<W> {
    fn drop(&mut self) {
        drop(self.receiver.take());
        for join in self.joins.drain(..) {
            let _ = join.join();
        }
    }
}

pub(crate) fn spawn_ordered_host_workers<W, F>(
    plans: Vec<HostWindowPlan>,
    worker_count: usize,
    prefetch_windows: usize,
    worker: F,
) -> OrderedHostWindowStream<W>
where
    W: Send + 'static,
    F: Fn(usize, HostWindowPlan) -> Result<W, LoaderWorkerError> + Send + Sync + 'static,
{
    let total_windows = plans.len();
    let plans = Arc::new(Mutex::new(VecDeque::from(plans)));
    let worker = Arc::new(worker);
    let (sender, receiver) = mpsc::sync_channel(prefetch_windows.max(1));
    let mut joins = Vec::with_capacity(worker_count);

    for worker_id in 0..worker_count {
        let plans = plans.clone();
        let worker = worker.clone();
        let sender = sender.clone();
        joins.push(thread::spawn(move || {
            loop {
                let plan = {
                    let mut plans = plans
                        .lock()
                        .expect("host-window plan queue should not be poisoned");
                    plans.pop_front()
                };
                let Some(plan) = plan else {
                    break;
                };
                let sequence = plan.sequence;
                let result = worker(worker_id, plan).map_err(|error| error.with_worker(worker_id));
                if sender.send(HostWindowMessage { sequence, result }).is_err() {
                    break;
                }
            }
        }));
    }
    drop(sender);

    OrderedHostWindowStream {
        receiver: Some(receiver),
        buffered: BTreeMap::new(),
        next_sequence: 0,
        total_windows,
        joins,
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct LoaderWindowProfile {
    pub(crate) wait: Duration,
    pub(crate) disk_read: Duration,
    pub(crate) host_pack: Duration,
    pub(crate) teacher_build: Duration,
    pub(crate) tensor_upload: Duration,
    pub(crate) producer_total: Duration,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct LoaderProfileAverage {
    pub(crate) windows: usize,
    pub(crate) wait_ms: f64,
    pub(crate) disk_read_ms: f64,
    pub(crate) host_pack_ms: f64,
    pub(crate) teacher_build_ms: f64,
    pub(crate) tensor_upload_ms: f64,
    pub(crate) producer_total_ms: f64,
}

#[derive(Debug)]
pub(crate) struct LoaderProfileAccumulator {
    label: String,
    every: usize,
    windows: usize,
    sum: LoaderWindowProfile,
    sink: LoaderProfileSink,
}

impl LoaderProfileAccumulator {
    pub(crate) fn new(label: impl Into<String>, every: usize) -> Self {
        Self {
            label: label.into(),
            every,
            windows: 0,
            sum: LoaderWindowProfile::default(),
            sink: LoaderProfileSink::from_env(every),
        }
    }

    pub(crate) fn record(&mut self, sample: LoaderWindowProfile) {
        if self.every == 0 {
            return;
        }
        self.windows += 1;
        self.sum.wait += sample.wait;
        self.sum.disk_read += sample.disk_read;
        self.sum.host_pack += sample.host_pack;
        self.sum.teacher_build += sample.teacher_build;
        self.sum.tensor_upload += sample.tensor_upload;
        self.sum.producer_total += sample.producer_total;

        if self.windows.is_multiple_of(self.every) {
            let average = self.average();
            self.sink.write_line(&format!(
                "{} loader profile: windows={} wait_ms={:.3} disk_read_ms={:.3} host_pack_ms={:.3} teacher_build_ms={:.3} tensor_upload_ms={:.3} producer_total_ms={:.3}",
                self.label,
                average.windows,
                average.wait_ms,
                average.disk_read_ms,
                average.host_pack_ms,
                average.teacher_build_ms,
                average.tensor_upload_ms,
                average.producer_total_ms,
            ));
            self.windows = 0;
            self.sum = LoaderWindowProfile::default();
        }
    }

    pub(crate) fn average(&self) -> LoaderProfileAverage {
        duration_average(self.sum, self.windows)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LoaderProfileSink {
    Disabled,
    Stderr,
    File(PathBuf),
}

impl LoaderProfileSink {
    fn from_env(every: usize) -> Self {
        if every == 0 {
            return Self::Disabled;
        }
        if let Some(value) = env::var_os("GEMS_LOADER_PROFILE_OUTPUT") {
            let value = value.to_string_lossy();
            let trimmed = value.trim();
            return match trimmed.to_ascii_lowercase().as_str() {
                "" | "hidden" | "hide" | "off" | "0" | "false" | "none" => Self::Disabled,
                "stderr" | "terminal" | "term" => Self::Stderr,
                _ => Self::File(PathBuf::from(trimmed)),
            };
        }
        if let Some(path) = env::var_os("GEMS_LOADER_PROFILE_LOG") {
            return Self::File(PathBuf::from(path));
        }
        if cfg!(feature = "tui") {
            if let Some(run_dir) = env::var_os("GEMS_RUN_DIR") {
                return Self::File(PathBuf::from(run_dir).join("loader-profile.log"));
            }
            return Self::File(PathBuf::from("loader-profile.log"));
        }
        Self::Stderr
    }

    fn write_line(&self, line: &str) {
        match self {
            Self::Disabled => {}
            Self::Stderr => eprintln!("{line}"),
            Self::File(path) => {
                if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty())
                    && fs::create_dir_all(parent).is_err()
                {
                    return;
                }
                let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
                    return;
                };
                let _ = writeln!(file, "{line}");
            }
        }
    }
}

fn duration_average(sum: LoaderWindowProfile, windows: usize) -> LoaderProfileAverage {
    if windows == 0 {
        return LoaderProfileAverage::default();
    }
    let divisor = windows as f64;
    LoaderProfileAverage {
        windows,
        wait_ms: sum.wait.as_secs_f64() * 1_000.0 / divisor,
        disk_read_ms: sum.disk_read.as_secs_f64() * 1_000.0 / divisor,
        host_pack_ms: sum.host_pack.as_secs_f64() * 1_000.0 / divisor,
        teacher_build_ms: sum.teacher_build.as_secs_f64() * 1_000.0 / divisor,
        tensor_upload_ms: sum.tensor_upload.as_secs_f64() * 1_000.0 / divisor,
        producer_total_ms: sum.producer_total.as_secs_f64() * 1_000.0 / divisor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_window_plan_covers_exact_epoch_items() {
        let plan = host_window_plan(10, 4);
        assert_eq!(
            plan,
            vec![
                HostWindowPlan {
                    sequence: 0,
                    start_item: 0,
                    items: 4
                },
                HostWindowPlan {
                    sequence: 1,
                    start_item: 4,
                    items: 4
                },
                HostWindowPlan {
                    sequence: 2,
                    start_item: 8,
                    items: 2
                }
            ]
        );
        let covered = plan.iter().map(|window| window.items).sum::<usize>();
        assert_eq!(covered, 10);
        assert_eq!(plan.first().map(|window| window.start_item), Some(0));
        assert_eq!(
            plan.last().map(|window| window.start_item + window.items),
            Some(10)
        );
    }

    #[test]
    fn profile_accumulator_averages_milliseconds() {
        let mut profile = LoaderProfileAccumulator::new("test", 10);
        profile.record(LoaderWindowProfile {
            wait: Duration::from_millis(1),
            disk_read: Duration::from_millis(2),
            host_pack: Duration::from_millis(3),
            teacher_build: Duration::from_millis(4),
            tensor_upload: Duration::from_millis(5),
            producer_total: Duration::from_millis(9),
        });
        profile.record(LoaderWindowProfile {
            wait: Duration::from_millis(3),
            disk_read: Duration::from_millis(4),
            host_pack: Duration::from_millis(5),
            teacher_build: Duration::from_millis(6),
            tensor_upload: Duration::from_millis(7),
            producer_total: Duration::from_millis(15),
        });

        let average = profile.average();
        assert_eq!(average.windows, 2);
        assert_eq!(average.wait_ms, 2.0);
        assert_eq!(average.disk_read_ms, 3.0);
        assert_eq!(average.host_pack_ms, 4.0);
        assert_eq!(average.teacher_build_ms, 5.0);
        assert_eq!(average.tensor_upload_ms, 6.0);
        assert_eq!(average.producer_total_ms, 12.0);
    }

    #[test]
    fn ordered_workers_preserve_plan_order() {
        let plans = host_window_plan(6, 2);
        let mut stream = spawn_ordered_host_workers(plans, 2, 2, |_worker_id, plan| {
            Ok((plan.sequence, plan.start_item))
        });

        assert_eq!(stream.next_window("test"), Some((0, 0)));
        assert_eq!(stream.next_window("test"), Some((1, 2)));
        assert_eq!(stream.next_window("test"), Some((2, 4)));
        assert_eq!(stream.next_window("test"), None);
    }
}
