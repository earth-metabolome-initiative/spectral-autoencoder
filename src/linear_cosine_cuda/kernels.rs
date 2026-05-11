use super::api::SIMILARITY_METRIC_MODIFIED_LINEAR_COSINE;
use burn_cubecl::cubecl::prelude::*;

#[cube(launch)]
pub(super) fn linear_cosine_preprocessed_paired_sorted_forward<F: Float>(
    left_mz: &Tensor<F>,
    left_intensity: &Tensor<F>,
    right_mz: &Tensor<F>,
    right_intensity: &Tensor<F>,
    mz_power: &Tensor<F>,
    intensity_power: &Tensor<F>,
    mz_tolerance: &Tensor<F>,
    output: &mut Tensor<F>,
    epsilon: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let right_rows = right_mz.shape(0);
    let row = ABSOLUTE_POS;

    if row >= right_rows {
        terminate!();
    }

    let tolerance = mz_tolerance[row * mz_tolerance.stride(0)];
    let eps = F::cast_from(epsilon);
    let mz_p = mz_power[row * mz_power.stride(0)];
    let intensity_p = intensity_power[row * intensity_power.stride(0)];

    output[ABSOLUTE_POS] = linear_cosine_score_rows(
        left_mz,
        left_intensity,
        row,
        right_mz,
        right_intensity,
        row,
        mz_p,
        intensity_p,
        tolerance,
        eps,
    );
}

#[cube(launch)]
pub(super) fn linear_cosine_similarity_ranking_forward<F: Float, I: Int>(
    teacher_mz: &Tensor<F>,
    teacher_intensity: &Tensor<F>,
    teacher_precursor: &Tensor<F>,
    candidate_index: &mut Tensor<I>,
    best_candidate_position: &mut Tensor<I>,
    top2_gap: &mut Tensor<F>,
    batch_start: u32,
    batch_items: u32,
    candidates_per_anchor: u32,
    mz_power: f32,
    intensity_power: f32,
    mz_tolerance: f32,
    seed: u32,
    epsilon: f32,
    #[comptime] metric: u32,
    #[comptime] max_peaks: usize,
) {
    if ABSOLUTE_POS >= best_candidate_position.len() {
        terminate!();
    }

    let anchor = ABSOLUTE_POS;
    if anchor >= batch_items as usize || batch_items < 3 {
        terminate!();
    }

    let candidate_count = candidates_per_anchor.max(2).min(batch_items - 1) as usize;
    let teacher_anchor = batch_start as usize + anchor;
    let mz_p = F::cast_from(mz_power);
    let intensity_p = F::cast_from(intensity_power);
    let tolerance = F::cast_from(mz_tolerance);
    let eps = F::cast_from(epsilon);
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let mut state = seed ^ (((anchor as u32) + 1u32) * 40503u32) ^ (batch_start >> 16);
    if state == 0u32 {
        state = 0x6d2b_79f5u32;
    }
    let partner_slots = batch_items - 1;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    let offset = state % partner_slots;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    let mut stride = (state % partner_slots) + 1u32;
    let mut coprime = false;
    while !coprime {
        let mut left = stride;
        let mut right = partner_slots;
        while right != 0u32 {
            let remainder = left % right;
            left = right;
            right = remainder;
        }
        coprime = left == 1u32;
        if !coprime {
            stride += 1u32;
            if stride > partner_slots {
                stride = 1u32;
            }
        }
    }

    let mut best_score = F::new(-1.0_f32);
    let mut second_best_score = F::new(-1.0_f32);
    let mut best_position = 0usize;

    for candidate_position in 0..candidate_count {
        let mut local_partner =
            ((offset + (candidate_position as u32) * stride) % partner_slots) as usize;
        if local_partner >= anchor {
            local_partner += 1;
        }

        candidate_index
            [anchor * candidate_index.stride(0) + candidate_position * candidate_index.stride(1)] =
            I::cast_from(local_partner as u32);

        let partner_row = batch_start as usize + local_partner;
        let score = if comptime!(metric == SIMILARITY_METRIC_MODIFIED_LINEAR_COSINE) {
            modified_linear_cosine_score_rows(
                teacher_mz,
                teacher_intensity,
                teacher_precursor,
                teacher_anchor,
                teacher_mz,
                teacher_intensity,
                teacher_precursor,
                partner_row,
                mz_p,
                intensity_p,
                tolerance,
                eps,
                max_peaks,
            )
        } else {
            linear_cosine_score_rows(
                teacher_mz,
                teacher_intensity,
                teacher_anchor,
                teacher_mz,
                teacher_intensity,
                partner_row,
                mz_p,
                intensity_p,
                tolerance,
                eps,
            )
        };

        if score > best_score {
            second_best_score = best_score;
            best_score = score;
            best_position = candidate_position;
        } else if score > second_best_score {
            second_best_score = score;
        }
    }

    best_candidate_position[anchor] = I::cast_from(best_position as u32);
    top2_gap[anchor] = (best_score - second_best_score).max(zero).min(one);
}

#[cube]
fn linear_cosine_score_rows<F: Float>(
    left_mz: &Tensor<F>,
    left_intensity: &Tensor<F>,
    left_row: usize,
    right_mz: &Tensor<F>,
    right_intensity: &Tensor<F>,
    right_row: usize,
    mz_p: F,
    intensity_p: F,
    tolerance: F,
    eps: F,
) -> F {
    let left_peaks = left_mz.shape(1);
    let right_peaks = right_mz.shape(1);
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);

    let mut left_intensity_max = zero;
    let mut left_mz_max = zero;
    for peak in 0..left_peaks {
        let intensity =
            left_intensity[left_row * left_intensity.stride(0) + peak * left_intensity.stride(1)];
        if intensity > zero {
            let mz = left_mz[left_row * left_mz.stride(0) + peak * left_mz.stride(1)];
            left_intensity_max = left_intensity_max.max(intensity.max(eps).powf(intensity_p));
            left_mz_max = left_mz_max.max(mz.max(eps).powf(mz_p));
        }
    }
    left_intensity_max += eps;
    left_mz_max += eps;

    let mut left_product_max = zero;
    let mut left_norm_square = zero;
    for peak in 0..left_peaks {
        let intensity =
            left_intensity[left_row * left_intensity.stride(0) + peak * left_intensity.stride(1)];
        if intensity > zero {
            let mz = left_mz[left_row * left_mz.stride(0) + peak * left_mz.stride(1)];
            let product = (intensity.max(eps).powf(intensity_p) / left_intensity_max)
                * (mz.max(eps).powf(mz_p) / left_mz_max);
            left_product_max = left_product_max.max(product);
        }
    }
    left_product_max += eps;
    for peak in 0..left_peaks {
        let product = peak_product(
            left_mz,
            left_intensity,
            left_row,
            peak,
            mz_p,
            intensity_p,
            left_mz_max,
            left_intensity_max,
            left_product_max,
            eps,
        );
        left_norm_square += product * product;
    }

    let mut right_intensity_max = zero;
    let mut right_mz_max = zero;
    for peak in 0..right_peaks {
        let intensity = right_intensity
            [right_row * right_intensity.stride(0) + peak * right_intensity.stride(1)];
        if intensity > zero {
            let mz = right_mz[right_row * right_mz.stride(0) + peak * right_mz.stride(1)];
            right_intensity_max = right_intensity_max.max(intensity.max(eps).powf(intensity_p));
            right_mz_max = right_mz_max.max(mz.max(eps).powf(mz_p));
        }
    }
    right_intensity_max += eps;
    right_mz_max += eps;

    let mut right_product_max = zero;
    let mut right_norm_square = zero;
    for peak in 0..right_peaks {
        let intensity = right_intensity
            [right_row * right_intensity.stride(0) + peak * right_intensity.stride(1)];
        if intensity > zero {
            let mz = right_mz[right_row * right_mz.stride(0) + peak * right_mz.stride(1)];
            let product = (intensity.max(eps).powf(intensity_p) / right_intensity_max)
                * (mz.max(eps).powf(mz_p) / right_mz_max);
            right_product_max = right_product_max.max(product);
        }
    }
    right_product_max += eps;
    for peak in 0..right_peaks {
        let product = peak_product(
            right_mz,
            right_intensity,
            right_row,
            peak,
            mz_p,
            intensity_p,
            right_mz_max,
            right_intensity_max,
            right_product_max,
            eps,
        );
        right_norm_square += product * product;
    }

    let mut left_cursor = 0usize;
    let mut right_cursor = 0usize;
    let mut score = zero;
    while left_cursor < left_peaks && right_cursor < right_peaks {
        let left_intensity_value = left_intensity
            [left_intensity.stride(0) * left_row + left_intensity.stride(1) * left_cursor];
        if left_intensity_value <= zero {
            left_cursor += 1;
        } else {
            let right_product = peak_product(
                right_mz,
                right_intensity,
                right_row,
                right_cursor,
                mz_p,
                intensity_p,
                right_mz_max,
                right_intensity_max,
                right_product_max,
                eps,
            );
            if right_product <= zero {
                right_cursor += 1;
            } else {
                let mz_left =
                    left_mz[left_mz.stride(0) * left_row + left_mz.stride(1) * left_cursor];
                let mz_right =
                    right_mz[right_mz.stride(0) * right_row + right_mz.stride(1) * right_cursor];
                let delta = mz_left - mz_right;

                if delta.abs() <= tolerance {
                    let left_product = peak_product(
                        left_mz,
                        left_intensity,
                        left_row,
                        left_cursor,
                        mz_p,
                        intensity_p,
                        left_mz_max,
                        left_intensity_max,
                        left_product_max,
                        eps,
                    );
                    score += left_product * right_product;
                    left_cursor += 1;
                    right_cursor += 1;
                } else if mz_left + tolerance < mz_right {
                    left_cursor += 1;
                } else {
                    right_cursor += 1;
                }
            }
        }
    }

    let left_norm = (left_norm_square + eps).sqrt();
    let right_norm = (right_norm_square + eps).sqrt();
    let similarity = score / (left_norm * right_norm + eps);
    similarity.max(zero).min(one)
}

#[cube]
pub(super) fn modified_linear_cosine_score_rows<F: Float>(
    left_mz: &Tensor<F>,
    left_intensity: &Tensor<F>,
    left_precursor: &Tensor<F>,
    left_row: usize,
    right_mz: &Tensor<F>,
    right_intensity: &Tensor<F>,
    right_precursor: &Tensor<F>,
    right_row: usize,
    mz_p: F,
    intensity_p: F,
    tolerance: F,
    eps: F,
    #[comptime] max_peaks: usize,
) -> F {
    let left_peaks = left_mz.shape(1);
    let right_peaks = right_mz.shape(1);
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let invalid = RuntimeCell::<u32>::new(4_294_967_295u32).read();
    let candidate_capacity = comptime!(max_peaks * 2usize);
    let dp_capacity = comptime!(max_peaks * 2usize + 1usize);

    let mut similarity = zero;

    if left_peaks <= max_peaks && right_peaks <= max_peaks {
        let mut left_products = Array::<F>::new(max_peaks);
        let mut right_products = Array::<F>::new(max_peaks);

        let mut left_mz_max = zero;
        let mut left_intensity_max = zero;
        for peak in 0..left_peaks {
            left_products[peak] = zero;
            let intensity = left_intensity
                [left_row * left_intensity.stride(0) + peak * left_intensity.stride(1)];
            if intensity > zero {
                let mz = left_mz[left_row * left_mz.stride(0) + peak * left_mz.stride(1)];
                left_mz_max = left_mz_max.max(mz.powf(mz_p));
                left_intensity_max = left_intensity_max.max(intensity.powf(intensity_p));
            }
        }

        let mut left_product_max = zero;
        for peak in 0..left_peaks {
            let intensity = left_intensity
                [left_row * left_intensity.stride(0) + peak * left_intensity.stride(1)];
            if intensity > zero {
                let mz = left_mz[left_row * left_mz.stride(0) + peak * left_mz.stride(1)];
                let mut mz_component = mz.powf(mz_p);
                if left_mz_max > zero {
                    mz_component /= left_mz_max;
                }
                let mut intensity_component = intensity.powf(intensity_p);
                if left_intensity_max > zero {
                    intensity_component /= left_intensity_max;
                }
                let product = mz_component * intensity_component;
                left_products[peak] = product;
                left_product_max = left_product_max.max(product);
            }
        }

        let mut left_norm_square = zero;
        for peak in 0..left_peaks {
            if left_product_max > zero {
                left_products[peak] /= left_product_max;
            }
            left_norm_square += left_products[peak] * left_products[peak];
        }

        let mut right_mz_max = zero;
        let mut right_intensity_max = zero;
        for peak in 0..right_peaks {
            right_products[peak] = zero;
            let intensity = right_intensity
                [right_row * right_intensity.stride(0) + peak * right_intensity.stride(1)];
            if intensity > zero {
                let mz = right_mz[right_row * right_mz.stride(0) + peak * right_mz.stride(1)];
                right_mz_max = right_mz_max.max(mz.powf(mz_p));
                right_intensity_max = right_intensity_max.max(intensity.powf(intensity_p));
            }
        }

        let mut right_product_max = zero;
        for peak in 0..right_peaks {
            let intensity = right_intensity
                [right_row * right_intensity.stride(0) + peak * right_intensity.stride(1)];
            if intensity > zero {
                let mz = right_mz[right_row * right_mz.stride(0) + peak * right_mz.stride(1)];
                let mut mz_component = mz.powf(mz_p);
                if right_mz_max > zero {
                    mz_component /= right_mz_max;
                }
                let mut intensity_component = intensity.powf(intensity_p);
                if right_intensity_max > zero {
                    intensity_component /= right_intensity_max;
                }
                let product = mz_component * intensity_component;
                right_products[peak] = product;
                right_product_max = right_product_max.max(product);
            }
        }

        let mut right_norm_square = zero;
        for peak in 0..right_peaks {
            if right_product_max > zero {
                right_products[peak] /= right_product_max;
            }
            right_norm_square += right_products[peak] * right_products[peak];
        }

        if left_norm_square > zero && right_norm_square > zero {
            let mut candidate_left = Array::<u32>::new(candidate_capacity);
            let mut candidate_right = Array::<u32>::new(candidate_capacity);
            let mut candidate_count = 0usize;

            let mut right_cursor = 0usize;
            for left_peak in 0..left_peaks {
                if left_products[left_peak] > zero {
                    let left_value =
                        left_mz[left_row * left_mz.stride(0) + left_peak * left_mz.stride(1)];
                    while right_cursor < right_peaks && right_products[right_cursor] <= zero {
                        right_cursor += 1;
                    }
                    while right_cursor < right_peaks {
                        if right_products[right_cursor] <= zero {
                            right_cursor += 1;
                        } else {
                            let right_value = right_mz[right_row * right_mz.stride(0)
                                + right_cursor * right_mz.stride(1)];
                            let delta = left_value - right_value;
                            if delta > tolerance {
                                right_cursor += 1;
                            } else if delta.abs() <= tolerance {
                                if candidate_count < candidate_capacity {
                                    candidate_left[candidate_count] = left_peak as u32;
                                    candidate_right[candidate_count] = right_cursor as u32;
                                    candidate_count += 1;
                                }
                                right_cursor += 1;
                            } else {
                                break;
                            }
                        }
                    }
                }
            }

            let left_precursor_value = left_precursor[left_row * left_precursor.stride(0)];
            let right_precursor_value = right_precursor[right_row * right_precursor.stride(0)];
            if right_precursor_value < left_precursor_value - tolerance
                || right_precursor_value > left_precursor_value + tolerance
            {
                let mut shifted_right_cursor = 0usize;
                for left_peak in 0..left_peaks {
                    if left_products[left_peak] > zero {
                        let left_value = left_mz
                            [left_row * left_mz.stride(0) + left_peak * left_mz.stride(1)]
                            - left_precursor_value;
                        while shifted_right_cursor < right_peaks
                            && right_products[shifted_right_cursor] <= zero
                        {
                            shifted_right_cursor += 1;
                        }
                        while shifted_right_cursor < right_peaks {
                            if right_products[shifted_right_cursor] <= zero {
                                shifted_right_cursor += 1;
                            } else {
                                let right_value = right_mz[right_row * right_mz.stride(0)
                                    + shifted_right_cursor * right_mz.stride(1)]
                                    - right_precursor_value;
                                let delta = left_value - right_value;
                                if delta > tolerance {
                                    shifted_right_cursor += 1;
                                } else if delta.abs() <= tolerance {
                                    if candidate_count < candidate_capacity {
                                        candidate_left[candidate_count] = left_peak as u32;
                                        candidate_right[candidate_count] =
                                            shifted_right_cursor as u32;
                                        candidate_count += 1;
                                    }
                                    shifted_right_cursor += 1;
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            if candidate_count > 0usize {
                for sort_index in 1..candidate_count {
                    let key_left = candidate_left[sort_index];
                    let key_right = candidate_right[sort_index];
                    let mut insert_index = sort_index;
                    while insert_index > 0usize {
                        let previous = insert_index - 1usize;
                        let previous_left = candidate_left[previous];
                        let previous_right = candidate_right[previous];
                        if previous_left > key_left
                            || (previous_left == key_left && previous_right > key_right)
                        {
                            candidate_left[insert_index] = previous_left;
                            candidate_right[insert_index] = previous_right;
                            insert_index -= 1;
                        } else {
                            break;
                        }
                    }
                    candidate_left[insert_index] = key_left;
                    candidate_right[insert_index] = key_right;
                }

                let mut unique_count = 0usize;
                for read_index in 0..candidate_count {
                    let current_left = candidate_left[read_index];
                    let current_right = candidate_right[read_index];
                    if read_index == 0usize
                        || current_left != candidate_left[read_index - 1usize]
                        || current_right != candidate_right[read_index - 1usize]
                    {
                        candidate_left[unique_count] = current_left;
                        candidate_right[unique_count] = current_right;
                        unique_count += 1;
                    }
                }
                candidate_count = unique_count;

                let mut left_slot_a = Array::<u32>::new(max_peaks);
                let mut left_slot_b = Array::<u32>::new(max_peaks);
                let mut right_slot_a = Array::<u32>::new(max_peaks);
                let mut right_slot_b = Array::<u32>::new(max_peaks);
                for peak in 0..max_peaks {
                    left_slot_a[peak] = invalid;
                    left_slot_b[peak] = invalid;
                    right_slot_a[peak] = invalid;
                    right_slot_b[peak] = invalid;
                }

                let mut neighbor_a = Array::<u32>::new(candidate_capacity);
                let mut neighbor_b = Array::<u32>::new(candidate_capacity);
                let mut visited = Array::<u32>::new(candidate_capacity);
                for edge in 0..candidate_count {
                    neighbor_a[edge] = invalid;
                    neighbor_b[edge] = invalid;
                    visited[edge] = 0u32;

                    let left_peak = candidate_left[edge] as usize;
                    if left_slot_a[left_peak] == invalid {
                        left_slot_a[left_peak] = edge as u32;
                    } else {
                        left_slot_b[left_peak] = edge as u32;
                    }

                    let right_peak = candidate_right[edge] as usize;
                    if right_slot_a[right_peak] == invalid {
                        right_slot_a[right_peak] = edge as u32;
                    } else {
                        right_slot_b[right_peak] = edge as u32;
                    }
                }

                for peak in 0..max_peaks {
                    let left_a = left_slot_a[peak];
                    let left_b = left_slot_b[peak];
                    if left_a != invalid && left_b != invalid {
                        insert_modified_neighbor(
                            &mut neighbor_a,
                            &mut neighbor_b,
                            left_a,
                            left_b,
                            invalid,
                        );
                        insert_modified_neighbor(
                            &mut neighbor_a,
                            &mut neighbor_b,
                            left_b,
                            left_a,
                            invalid,
                        );
                    }

                    let right_a = right_slot_a[peak];
                    let right_b = right_slot_b[peak];
                    if right_a != invalid && right_b != invalid {
                        insert_modified_neighbor(
                            &mut neighbor_a,
                            &mut neighbor_b,
                            right_a,
                            right_b,
                            invalid,
                        );
                        insert_modified_neighbor(
                            &mut neighbor_a,
                            &mut neighbor_b,
                            right_b,
                            right_a,
                            invalid,
                        );
                    }
                }

                let mut path = Array::<u32>::new(candidate_capacity);
                let mut benefits = Array::<F>::new(candidate_capacity);
                let mut dp = Array::<F>::new(dp_capacity);
                let mut score = zero;

                for start in 0..candidate_count {
                    if visited[start] == 0u32 {
                        let mut end = start;
                        let mut from = invalid;
                        loop {
                            let next = first_modified_neighbor_not_from(
                                neighbor_a[end],
                                neighbor_b[end],
                                from,
                                invalid,
                            );
                            if next == invalid {
                                break;
                            }
                            from = end as u32;
                            end = next as usize;
                        }

                        let mut path_len = 0usize;
                        let mut current = end;
                        let mut previous = invalid;
                        loop {
                            visited[current] = 1u32;
                            path[path_len] = current as u32;
                            path_len += 1;

                            let next = first_modified_neighbor_not_from(
                                neighbor_a[current],
                                neighbor_b[current],
                                previous,
                                invalid,
                            );
                            if next == invalid {
                                break;
                            }
                            previous = current as u32;
                            current = next as usize;
                        }

                        if path_len == 1usize {
                            let edge = path[0] as usize;
                            let left_peak = candidate_left[edge] as usize;
                            let right_peak = candidate_right[edge] as usize;
                            score += left_products[left_peak] * right_products[right_peak];
                        } else {
                            for path_index in 0..path_len {
                                let edge = path[path_index] as usize;
                                let left_peak = candidate_left[edge] as usize;
                                let right_peak = candidate_right[edge] as usize;
                                benefits[path_index] = (left_products[left_peak]
                                    * right_products[right_peak])
                                    .max(eps);
                            }

                            dp[0] = zero;
                            dp[1] = benefits[0];
                            for index in 2..(path_len + 1usize) {
                                let take = dp[index - 2usize] + benefits[index - 1usize];
                                let skip = dp[index - 1usize];
                                dp[index] = if take >= skip { take } else { skip };
                            }

                            let mut index = path_len;
                            while index > 0usize {
                                if index == 1usize {
                                    let edge = path[0] as usize;
                                    let left_peak = candidate_left[edge] as usize;
                                    let right_peak = candidate_right[edge] as usize;
                                    score += left_products[left_peak] * right_products[right_peak];
                                    break;
                                }
                                let take = dp[index - 2usize] + benefits[index - 1usize];
                                if take >= dp[index - 1usize] {
                                    let edge = path[index - 1usize] as usize;
                                    let left_peak = candidate_left[edge] as usize;
                                    let right_peak = candidate_right[edge] as usize;
                                    score += left_products[left_peak] * right_products[right_peak];
                                    index -= 2usize;
                                } else {
                                    index -= 1usize;
                                }
                            }
                        }
                    }
                }

                similarity = (score / ((left_norm_square.sqrt() * right_norm_square.sqrt()) + eps))
                    .max(zero)
                    .min(one);
            }
        }
    }
    similarity
}

#[cube]
fn insert_modified_neighbor(
    neighbor_a: &mut Array<u32>,
    neighbor_b: &mut Array<u32>,
    edge: u32,
    neighbor: u32,
    invalid: u32,
) {
    let edge_index = edge as usize;
    if neighbor_a[edge_index] == invalid {
        neighbor_a[edge_index] = neighbor;
    } else if neighbor_a[edge_index] != neighbor {
        neighbor_b[edge_index] = neighbor;
    }
}

#[cube]
fn first_modified_neighbor_not_from(
    neighbor_a: u32,
    neighbor_b: u32,
    from: u32,
    invalid: u32,
) -> u32 {
    let mut next = invalid;
    if neighbor_a != invalid && neighbor_a != from {
        next = neighbor_a;
    } else if neighbor_b != invalid && neighbor_b != from {
        next = neighbor_b;
    }
    next
}

#[cube]
fn peak_product<F: Float>(
    mz_tensor: &Tensor<F>,
    intensity_tensor: &Tensor<F>,
    row: usize,
    peak: usize,
    mz_power: F,
    intensity_power: F,
    mz_max: F,
    intensity_max: F,
    product_max: F,
    epsilon: F,
) -> F {
    let zero = F::new(0.0_f32);
    let intensity =
        intensity_tensor[row * intensity_tensor.stride(0) + peak * intensity_tensor.stride(1)];
    let mut product = zero;
    if intensity > zero {
        let mz = mz_tensor[row * mz_tensor.stride(0) + peak * mz_tensor.stride(1)];
        let intensity_component = intensity.max(epsilon).powf(intensity_power) / intensity_max;
        let mz_component = mz.max(epsilon).powf(mz_power) / mz_max;

        product = intensity_component * mz_component / product_max;
    }

    product
}
