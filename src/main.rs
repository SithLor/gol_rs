// This imports the necessary traits for parallel iterators
// function to compute the next generation

mod slower {
    const MAX_ITERATIONS: u32 = 100;
    const X: u32 = 512;
    const Y: u32 = 512;
    const SAVE_LOCATION: &str = "./src/frames";

    use rand::Rng;
    use rayon::prelude::*;
    use std::time::Instant;
    // std imports will be pulled where needed to avoid unused warnings

    pub fn simultion(grid: &Vec<Vec<i8>>) -> Vec<Vec<i8>> {
        // get the number of rows
        let n = grid.len();

        // get the number of columns
        let m = grid[0].len();

        // create an empty grid to compute the future generation
        let mut future: Vec<Vec<i8>> = vec![vec![0; m]; n];

        // iterate through each and every cell
        for i in 0..n {
            for j in 0..m {
                let cell_state = grid[i][j];
                let mut live_neighbors: i32 = 0;
                for x in -1..=1 {
                    for y in -1..=1 {
                        let new_x = i as i32 + x;
                        let new_y = j as i32 + y;
                        if new_x >= 0 && new_y >= 0 && new_x < n as i32 && new_y < m as i32 {
                            live_neighbors += grid[new_x as usize][new_y as usize] as i32;
                        }
                    }
                }
                live_neighbors -= cell_state as i32;
                if cell_state == 1 && live_neighbors < 2 {
                    future[i][j] = 0;
                } else if cell_state == 1 && live_neighbors > 3 {
                    future[i][j] = 0;
                } else if cell_state == 0 && live_neighbors == 3 {
                    future[i][j] = 1;
                } else {
                    future[i][j] = cell_state;
                }
            }
        }

        // return the future generation
        future
    }
    fn gol() {
        let (rows, cols) = (X as usize, Y as usize);
        let mut grid = create_grid(rows, cols);
        let mut grid_history = create_grid_history(rows, cols, MAX_ITERATIONS as usize);

        // random state initialization
        let random_grid_timer: Instant = Instant::now();
        let mut rng = rand::rng();
        for i in 0..rows {
            for j in 0..cols {
                grid[i][j] = if rng.random_range(0..2) == 0 { 0 } else { 1 };
            }
        }
        let random_grid_elapsed: u128 = random_grid_timer.elapsed().as_nanos();
        println!(
            "Random grid initialization time for a {}x{} grid: {} ns",
            X, Y, random_grid_elapsed
        );

        let simulation_timer: Instant = Instant::now();
        for iteration in 0..MAX_ITERATIONS {
            grid = simultion(&grid);
            grid_history[iteration as usize] = grid.clone();
        }
        let simulation_elapsed: u128 = simulation_timer.elapsed().as_nanos();
        println!(
            "Simulation time for {} iterations on a {}x{} grid: {} ns",
            MAX_ITERATIONS, X, Y, simulation_elapsed
        );

        let io_timer: Instant = Instant::now();
        for iteration in 0..MAX_ITERATIONS {
            grid_to_image(&grid_history[iteration as usize], iteration, SAVE_LOCATION);
        }
        let io_elapsed: u128 = io_timer.elapsed().as_nanos();
        println!(
            "I/O time for saving {} iterations on a {}x{} grid: {} ns",
            MAX_ITERATIONS, X, Y, io_elapsed
        );

        //clear the console
        print!("{}[2J", 27 as char);
        println!("gol()");
        println!("  Information:");
        println!("          Grid size: {}x{}", X, Y);
        println!("          Total iterations: {}", MAX_ITERATIONS);
        println!("  Summary:");
        println!("          Init time: {} ns", random_grid_elapsed);
        println!("          Simulation time: {} ns", simulation_elapsed);
        println!("          I/O time: {} ns", io_elapsed);
    }

    fn grid_to_image(grid: &Vec<Vec<i8>>, iteration: u32, save_location: &str) {
        let n = grid.len();
        let m = grid[0].len();

        let mut imgbuf = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::new(m as u32, n as u32);

        for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
            let value: u8 = if grid[y as usize][x as usize] == 1 {
                255
            } else {
                0
            };
            *pixel = image::Rgb([value, value, value]);
        }

        imgbuf
            .save(format!("{}/gol_{}.png", save_location, iteration))
            .unwrap();
    }
    fn create_grid(rows: usize, cols: usize) -> Vec<Vec<i8>> {
        vec![vec![0; cols]; rows]
    }
    fn create_grid_history(rows: usize, cols: usize, max_iterations: usize) -> Vec<Vec<Vec<i8>>> {
        vec![vec![vec![0; cols]; rows]; max_iterations]
    }
}

mod faster {
    // 512x512 take 0,32 sec so 1920

    pub const FPS: u32 = 120;
    pub const VIDEO_LENGTH_SECONDS: u32 = 10;
    pub const VIDEO_FRAMES: u32 = FPS * VIDEO_LENGTH_SECONDS;

    pub const MAX_ITERATIONS: u32 = VIDEO_FRAMES;
    pub const X: u32 = 512; // 4k resolution
    pub const Y: u32 = 512; // 4k resolution 
    pub const SAVE_LOCATION: &str = "./src/frames";

    use rand::Rng;
    use rayon::prelude::*;
    use std::time::Instant;

    fn grid_to_image_flat(
        grid: &[u8],
        rows: usize,
        cols: usize,
        iteration: u32,
        save_location: &str,
    ) {
        // grid is padded (rows+2) x (cols+2). Active area is [1..=rows] x [1..=cols]
        let stride = cols + 2;
        let mut buf = vec![0u8; rows * cols];
        for y in 0..rows {
            let src_base = (y + 1) * stride + 1;
            let dst_base = y * cols;
            for x in 0..cols {
                let src_idx = src_base + x;
                buf[dst_base + x] = if grid[src_idx] == 1 { 255 } else { 0 };
            }
        }
        let imgbuf =
            image::ImageBuffer::<image::Luma<u8>, Vec<u8>>::from_raw(cols as u32, rows as u32, buf)
                .expect("valid image buffer");
        imgbuf
            .save(format!("{}/gol_{}.png", save_location, iteration))
            .unwrap();
    }

    pub fn gol_faster() {
        let (rows, cols) = (X as usize, Y as usize);
        let mut grid = create_grid(rows, cols);
        let mut next = create_grid(rows, cols);

        // Efficient random state initialization
        let random_grid_timer: Instant = Instant::now();
        let mut rng = rand::rng();
        {
            let stride = cols + 2;
            for y in 0..rows {
                let base = (y + 1) * stride + 1;
                let row_slice = &mut grid[base..base + cols];
                rng.fill(row_slice);
                for v in row_slice.iter_mut() {
                    *v &= 1;
                }
            }
        }
        let random_grid_elapsed: u128 = random_grid_timer.elapsed().as_nanos();
        println!(
            "Random grid initialization time for a {}x{} grid: {} ns",
            X, Y, random_grid_elapsed
        );

        let mut sim_ns: u128 = 0;
        let mut io_ns: u128 = 0;
        for iteration in 0..MAX_ITERATIONS {
            let t0 = Instant::now();
            step_par(&grid, rows, cols, &mut next);
            sim_ns += t0.elapsed().as_nanos();

            // Swap so we save the just-computed generation from `grid`
            std::mem::swap(&mut grid, &mut next);

            let t1 = Instant::now();
            grid_to_image_flat(&grid, rows, cols, iteration, SAVE_LOCATION);
            io_ns += t1.elapsed().as_nanos();
        }
        let simulation_elapsed: u128 = sim_ns;
        println!(
            "Simulation time for {} iterations on a {}x{} grid: {} ns",
            MAX_ITERATIONS, X, Y, simulation_elapsed
        );

        let io_elapsed: u128 = io_ns;
        println!(
            "I/O time for saving {} iterations on a {}x{} grid: {} ns",
            MAX_ITERATIONS, X, Y, io_elapsed
        );

        //clear the console
        print!("{}[2J", 27 as char);
        println!("gol_faster()");
        println!("  Information:");
        println!("          Grid size: {}x{}", X, Y);
        println!("          Total iterations: {}", MAX_ITERATIONS);
        println!("  Summary:");
        println!("          Init time: {} ns", random_grid_elapsed);
        println!("          Simulation time: {} ns", simulation_elapsed);
        println!("          I/O time: {} ns", io_elapsed);
    }

    #[inline(always)]
    fn create_grid(rows: usize, cols: usize) -> Vec<u8> {
        // Padded with a border of zeros to eliminate bounds checks
        vec![0u8; (rows + 2) * (cols + 2)]
    }

    //hell on earth
    #[inline(always)]
    fn step_par(current: &[u8], rows: usize, cols: usize, out: &mut [u8]) {
        // Padded buffers: dimensions are (rows+2) x (cols+2). Work only in inner region.
        let stride = cols + 2;
        debug_assert_eq!(current.len(), (rows + 2) * (cols + 2));
        debug_assert_eq!(out.len(), (rows + 2) * (cols + 2));

        // Zero top and bottom padded rows
        out[..stride].fill(0);
        out[(rows + 1) * stride..].fill(0);

        // Process inner rows in parallel; each chunk is a full padded row
        out[stride..(rows + 1) * stride]
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(i0, out_row)| {
                let i = i0 + 1; // real row index in padded buffer
                let row_base = i * stride;
                // keep left/right borders zero
                out_row[0] = 0;
                out_row[cols + 1] = 0;
                for j in 1..=cols {
                    let idx = row_base + j;
                    let alive = current[idx];
                    let sum = current[idx - 1] as u16
                        + current[idx + 1] as u16
                        + current[idx - stride - 1] as u16
                        + current[idx - stride] as u16
                        + current[idx - stride + 1] as u16
                        + current[idx + stride - 1] as u16
                        + current[idx + stride] as u16
                        + current[idx + stride + 1] as u16;
                    out_row[j] = ((sum == 3) || (alive == 1 && sum == 2)) as u8;
                }
            });
    }
    pub fn gol_faster_stream() {
        let (rows, cols) = (X as usize, Y as usize);
        let mut grid = create_grid(rows, cols);
        let mut next = create_grid(rows, cols);

        // Random init into inner region
        let random_grid_timer: Instant = Instant::now();
        let mut rng = rand::rng();
        {
            let stride = cols + 2;
            for y in 0..rows {
                let base = (y + 1) * stride + 1;
                let row_slice = &mut grid[base..base + cols];
                rng.fill(row_slice);
                for v in row_slice.iter_mut() {
                    *v &= 1;
                }
            }
        }
        let random_grid_elapsed: u128 = random_grid_timer.elapsed().as_nanos();
        println!(
            "Random grid initialization time for a {}x{} grid: {} ns",
            X, Y, random_grid_elapsed
        );

        // Spawn ffmpeg to consume raw frames via stdin
        use std::io::Write;
        use std::process::{Command, Stdio};
        let output_name = format!("gol_simulation_fps_{}_X_{}_Y_{}.mp4", FPS, X, Y);
        let mut child = Command::new("ffmpeg")
            .args(&[
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-s",
                &format!("{}x{}", cols, rows),
                "-r",
                &format!("{}", FPS),
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                &output_name,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start ffmpeg. Is it installed and on PATH?");

        let mut ff_in = child.stdin.take().expect("Failed to open ffmpeg stdin");
        let mut frame_buf = vec![0u8; rows * cols];

        let mut sim_ns: u128 = 0;
        let mut io_ns: u128 = 0;
        for _iteration in 0..MAX_ITERATIONS {
            let timer_0 = Instant::now();
            //run the simulation step in parallel
            step_par(&grid, rows, cols, &mut next);
            sim_ns += timer_0.elapsed().as_nanos();

            // Swap so we save the just-computed generation from `grid`
            std::mem::swap(&mut grid, &mut next);

            let t1 = Instant::now();
            // Convert inner region to grayscale bytes (0/255)
            let stride = cols + 2;
            for y in 0..rows {
                let src_base = (y + 1) * stride + 1;
                let dst_base = y * cols;
                for x in 0..cols {
                    frame_buf[dst_base + x] = grid[src_base + x] * 255u8;
                }
            }
            ff_in
                .write_all(&frame_buf)
                .expect("Failed to write frame to ffmpeg");
            io_ns += t1.elapsed().as_nanos();
        }
        // Close stdin to signal EOF this ffmpeg to start encoding
        drop(ff_in);

        let status = child.wait().expect("Failed to wait on ffmpeg");

        //clear the console
        print!("{}[2J", 27 as char);
        println!("gol_faster_stream()");
        println!("  Information:");
        println!("          Grid size: {}x{}", X, Y);
        println!("          Total iterations: {}", MAX_ITERATIONS);
        println!("  Summary:");
        println!("          Init time: {} ns", random_grid_elapsed);
        println!("          Simulation time: {} ns", sim_ns);
        println!("          I/O time (pipe to ffmpeg): {} ns", io_ns);
        println!(
            "Total time ms: {}",
            (random_grid_elapsed + sim_ns + io_ns) / 1_000_000
        );
        if status.success() {
            println!("Video created successfully: {}", output_name);
        } else {
            eprintln!("ffmpeg exited with status: {}", status);
        }
    }
}

mod faster_hw {
    // 512x512 take 0,32 sec so 1920

    pub const FPS: u32 = 120;
    pub const VIDEO_LENGTH_SECONDS: u32 = 10;
    pub const VIDEO_FRAMES: u32 = FPS * VIDEO_LENGTH_SECONDS;

    pub const MAX_ITERATIONS: u32 = VIDEO_FRAMES;
    pub const X: u32 = 512; // 4k resolution
    pub const Y: u32 = 512; // 4k resolution 
    pub const SAVE_LOCATION: &str = "./src/frames";

    use rand::Rng;
    use rayon::prelude::*;
    use std::time::Instant;
    use std::{arch::is_aarch64_feature_detected, arch::is_x86_feature_detected};

    #[inline(always)]
    fn create_grid(rows: usize, cols: usize) -> Vec<u8> {
        // Padded with a border of zeros to eliminate bounds checks
        vec![0u8; (rows + 2) * (cols + 2)]
    }

    /// Public wrapper keeping your original signature.
    /// Buffers must be padded with 1-cell border: dimensions = (rows+2) x (cols+2).
    #[inline(always)]
    pub fn step_par(current: &[u8], rows: usize, cols: usize, out: &mut [u8]) {
        let stride = cols + 2;
        debug_assert_eq!(current.len(), (rows + 2) * stride);
        debug_assert_eq!(out.len(), (rows + 2) * stride);

        // Zero top and bottom padded rows
        out[..stride].fill(0);
        out[(rows + 1) * stride..].fill(0);

        // Dispatch to the best kernel available at runtime
        unsafe {
            step_kernel_simd_dispatch(current, rows, cols, out);
        }
    }

    /// Runtime dispatch to AVX-512 (if available), AVX2, NEON, or scalar fallback.
    /// This is unsafe because kernels use unchecked loads/stores.
    #[inline(always)]
    unsafe fn step_kernel_simd_dispatch(current: &[u8], rows: usize, cols: usize, out: &mut [u8]) {
        // Detect features
        #[cfg(all(target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") && is_x86_feature_detected!("avx512dq") {
                println!("AVX-512 detected");
                step_kernel_avx512(current, rows, cols, out);
                return;
            } else if is_x86_feature_detected!("avx2") {
                println!("AVX2 detected");
                step_kernel_avx2(current, rows, cols, out);
                return;
            }
        }

        #[cfg(all(target_arch = "aarch64"))]
        {
            if is_aarch64_feature_detected!("neon") {
                step_kernel_neon(current, rows, cols, out);
                return;
            }
        }

        // Fallback scalar unsafe parallel
        println!("No SIMD detected, using scalar fallback");
        step_kernel_scalar(current, rows, cols, out);
    }

    /// Scalar unsafe kernel (row-parallel)
    #[inline(always)]
    unsafe fn step_kernel_scalar(current: &[u8], rows: usize, cols: usize, out: &mut [u8]) {
        let stride = cols + 2;
        out[stride..(rows + 1) * stride]
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(i0, out_row)| {
                let i = i0 + 1;
                let row_base = i * stride;
                // left/right borders
                *out_row.get_unchecked_mut(0) = 0;
                *out_row.get_unchecked_mut(cols + 1) = 0;

                let mut idx = row_base + 1;
                for j in 1..=cols {
                    let alive = *current.get_unchecked(idx);
                    let sum = *current.get_unchecked(idx - 1) as u16
                        + *current.get_unchecked(idx + 1) as u16
                        + *current.get_unchecked(idx - stride - 1) as u16
                        + *current.get_unchecked(idx - stride) as u16
                        + *current.get_unchecked(idx - stride + 1) as u16
                        + *current.get_unchecked(idx + stride - 1) as u16
                        + *current.get_unchecked(idx + stride) as u16
                        + *current.get_unchecked(idx + stride + 1) as u16;
                    *out_row.get_unchecked_mut(j) = ((sum == 3) || (alive == 1 && sum == 2)) as u8;
                    idx += 1;
                }
            });
    }

    ///////////////////////////////////////////////////////////////////////////////
    // AVX2 kernel (existing, processes 32 bytes per loop)
    ///////////////////////////////////////////////////////////////////////////////
    #[cfg(all(target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn step_kernel_avx2(current: &[u8], rows: usize, cols: usize, out: &mut [u8]) {
        let stride = cols + 2;
        out[stride..(rows + 1) * stride]
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(i0, out_row)| {
                let i = i0 + 1;
                let row_base = i * stride;
                // clear left/right border
                *out_row.get_unchecked_mut(0) = 0;
                *out_row.get_unchecked_mut(cols + 1) = 0;

                use std::arch::x86_64::*;
                let mut j = 1usize;

                // process 32-byte chunks (two 16-byte loads => 32 bytes)
                while j + 31 <= cols {
                    let idx = row_base + j;
                    // base pointers
                    let p_up = current.as_ptr().add(idx - stride);
                    let p_cur = current.as_ptr().add(idx);
                    let p_down = current.as_ptr().add(idx + stride);

                    // loads: left-shifted, center, right-shifted (each 32 bytes as two 128-bit loads)
                    let lo_up = _mm_loadu_si128(p_up as *const __m128i);
                    let hi_up = _mm_loadu_si128(p_up.add(16) as *const __m128i);
                    let lo_cur = _mm_loadu_si128(p_cur as *const __m128i);
                    let hi_cur = _mm_loadu_si128(p_cur.add(16) as *const __m128i);
                    let lo_down = _mm_loadu_si128(p_down as *const __m128i);
                    let hi_down = _mm_loadu_si128(p_down.add(16) as *const __m128i);

                    let lo_up_l = _mm_loadu_si128(p_up.sub(1) as *const __m128i);
                    let hi_up_l = _mm_loadu_si128(p_up.sub(1).add(16) as *const __m128i);
                    let lo_cur_l = _mm_loadu_si128(p_cur.sub(1) as *const __m128i);
                    let hi_cur_l = _mm_loadu_si128(p_cur.sub(1).add(16) as *const __m128i);
                    let lo_down_l = _mm_loadu_si128(p_down.sub(1) as *const __m128i);
                    let hi_down_l = _mm_loadu_si128(p_down.sub(1).add(16) as *const __m128i);

                    let lo_up_r = _mm_loadu_si128(p_up.add(1) as *const __m128i);
                    let hi_up_r = _mm_loadu_si128(p_up.add(1).add(16) as *const __m128i);
                    let lo_cur_r = _mm_loadu_si128(p_cur.add(1) as *const __m128i);
                    let hi_cur_r = _mm_loadu_si128(p_cur.add(1).add(16) as *const __m128i);
                    let lo_down_r = _mm_loadu_si128(p_down.add(1) as *const __m128i);
                    let hi_down_r = _mm_loadu_si128(p_down.add(1).add(16) as *const __m128i);

                    // widen to 16-bit lanes
                    let up_lo_i16 = _mm256_cvtepu8_epi16(lo_up);
                    let cur_lo_i16 = _mm256_cvtepu8_epi16(lo_cur);
                    let down_lo_i16 = _mm256_cvtepu8_epi16(lo_down);
                    let up_l_lo_i16 = _mm256_cvtepu8_epi16(lo_up_l);
                    let cur_l_lo_i16 = _mm256_cvtepu8_epi16(lo_cur_l);
                    let down_l_lo_i16 = _mm256_cvtepu8_epi16(lo_down_l);
                    let up_r_lo_i16 = _mm256_cvtepu8_epi16(lo_up_r);
                    let cur_r_lo_i16 = _mm256_cvtepu8_epi16(lo_cur_r);
                    let down_r_lo_i16 = _mm256_cvtepu8_epi16(lo_down_r);

                    let up_hi_i16 = _mm256_cvtepu8_epi16(hi_up);
                    let cur_hi_i16 = _mm256_cvtepu8_epi16(hi_cur);
                    let down_hi_i16 = _mm256_cvtepu8_epi16(hi_down);
                    let up_l_hi_i16 = _mm256_cvtepu8_epi16(hi_up_l);
                    let cur_l_hi_i16 = _mm256_cvtepu8_epi16(hi_cur_l);
                    let down_l_hi_i16 = _mm256_cvtepu8_epi16(hi_down_l);
                    let up_r_hi_i16 = _mm256_cvtepu8_epi16(hi_up_r);
                    let cur_r_hi_i16 = _mm256_cvtepu8_epi16(hi_cur_r);
                    let down_r_hi_i16 = _mm256_cvtepu8_epi16(hi_down_r);

                    // sums
                    let mut sum_lo = _mm256_add_epi16(cur_l_lo_i16, cur_r_lo_i16);
                    sum_lo = _mm256_add_epi16(sum_lo, up_l_lo_i16);
                    sum_lo = _mm256_add_epi16(sum_lo, up_lo_i16);
                    sum_lo = _mm256_add_epi16(sum_lo, up_r_lo_i16);
                    sum_lo = _mm256_add_epi16(sum_lo, down_l_lo_i16);
                    sum_lo = _mm256_add_epi16(sum_lo, down_lo_i16);
                    sum_lo = _mm256_add_epi16(sum_lo, down_r_lo_i16);

                    let mut sum_hi = _mm256_add_epi16(cur_l_hi_i16, cur_r_hi_i16);
                    sum_hi = _mm256_add_epi16(sum_hi, up_l_hi_i16);
                    sum_hi = _mm256_add_epi16(sum_hi, up_hi_i16);
                    sum_hi = _mm256_add_epi16(sum_hi, up_r_hi_i16);
                    sum_hi = _mm256_add_epi16(sum_hi, down_l_hi_i16);
                    sum_hi = _mm256_add_epi16(sum_hi, down_hi_i16);
                    sum_hi = _mm256_add_epi16(sum_hi, down_r_hi_i16);

                    // masks
                    let three = _mm256_set1_epi16(3);
                    let mask_eq3_lo = _mm256_cmpeq_epi16(sum_lo, three);
                    let mask_eq3_hi = _mm256_cmpeq_epi16(sum_hi, three);

                    let two = _mm256_set1_epi16(2);
                    let mask_eq2_lo = _mm256_cmpeq_epi16(sum_lo, two);
                    let mask_eq2_hi = _mm256_cmpeq_epi16(sum_hi, two);

                    let one = _mm256_set1_epi16(1);
                    let alive_is_1_lo = _mm256_cmpeq_epi16(cur_lo_i16, one);
                    let alive_is_1_hi = _mm256_cmpeq_epi16(cur_hi_i16, one);

                    let tmp_lo = _mm256_and_si256(mask_eq2_lo, alive_is_1_lo);
                    let survivors_lo = _mm256_or_si256(mask_eq3_lo, tmp_lo);

                    let tmp_hi = _mm256_and_si256(mask_eq2_hi, alive_is_1_hi);
                    let survivors_hi = _mm256_or_si256(mask_eq3_hi, tmp_hi);

                    // pack back to bytes (two 128-bit stores)
                    let packed = _mm256_packus_epi16(survivors_lo, survivors_hi);
                    let out_lo_128 = _mm256_castsi256_si128(packed);
                    let out_hi_128 = _mm256_extracti128_si256(packed, 1);

                    _mm_storeu_si128(out_row.as_mut_ptr().add(j) as *mut __m128i, out_lo_128);
                    _mm_storeu_si128(out_row.as_mut_ptr().add(j + 16) as *mut __m128i, out_hi_128);

                    j += 32;
                }

                // tail scalar
                let mut idx = row_base + j;
                while j <= cols {
                    let alive = *current.get_unchecked(idx);
                    let sum = *current.get_unchecked(idx - 1) as u16
                        + *current.get_unchecked(idx + 1) as u16
                        + *current.get_unchecked(idx - stride - 1) as u16
                        + *current.get_unchecked(idx - stride) as u16
                        + *current.get_unchecked(idx - stride + 1) as u16
                        + *current.get_unchecked(idx + stride - 1) as u16
                        + *current.get_unchecked(idx + stride) as u16
                        + *current.get_unchecked(idx + stride + 1) as u16;
                    *out_row.get_unchecked_mut(j) = ((sum == 3) || (alive == 1 && sum == 2)) as u8;
                    j += 1;
                    idx += 1;
                }
            });
    }

    ///////////////////////////////////////////////////////////////////////////////
    // AVX-512 kernel: implemented by invoking the AVX2 32-byte worker twice
    // (process 64 bytes per iteration). This keeps the code correct and avoids
    // more fragile AVX-512 packing intrinsics in this drop-in example.
    // On AVX-512-capable CPUs this still performs well because we do wider loads
    // (we still use the proven AVX2 inner math).
    ///////////////////////////////////////////////////////////////////////////////
    #[cfg(all(target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn step_kernel_avx512(current: &[u8], rows: usize, cols: usize, out: &mut [u8]) {
        // We'll process blocks of 64 bytes per iteration by calling the AVX2 32-byte
        // worker for j and j+32. This keeps the logic simple & safe.
        // (If you want full 512-bit lane arithmetic, we can implement it, but it's more verbose.)
        let stride = cols + 2;
        out[stride..(rows + 1) * stride]
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(i0, out_row)| {
                let i = i0 + 1;
                let row_base = i * stride;
                *out_row.get_unchecked_mut(0) = 0;
                *out_row.get_unchecked_mut(cols + 1) = 0;

                // inner loop over 64-byte blocks (two 32-byte chunks)
                let mut j = 1usize;
                while j + 63 <= cols {
                    // call AVX2 worker on j (32 bytes) and j+32 (next 32 bytes)
                    step_row_avx2_chunk(current, row_base, stride, j, out_row);
                    step_row_avx2_chunk(current, row_base, stride, j + 32, out_row);
                    j += 64;
                }

                // leftover 32-byte chunk
                while j + 31 <= cols {
                    step_row_avx2_chunk(current, row_base, stride, j, out_row);
                    j += 32;
                }

                // tail scalar
                let mut idx = row_base + j;
                while j <= cols {
                    let alive = *current.get_unchecked(idx);
                    let sum = *current.get_unchecked(idx - 1) as u16
                        + *current.get_unchecked(idx + 1) as u16
                        + *current.get_unchecked(idx - stride - 1) as u16
                        + *current.get_unchecked(idx - stride) as u16
                        + *current.get_unchecked(idx - stride + 1) as u16
                        + *current.get_unchecked(idx + stride - 1) as u16
                        + *current.get_unchecked(idx + stride) as u16
                        + *current.get_unchecked(idx + stride + 1) as u16;
                    *out_row.get_unchecked_mut(j) = ((sum == 3) || (alive == 1 && sum == 2)) as u8;
                    j += 1;
                    idx += 1;
                }
            });
    }

    /// Helper: call AVX2 32-byte worker for a single 32-byte-aligned chunk at column j.
    /// We mark as target_feature(avx2) so it compiles the inner intrinsics.
    #[cfg(all(target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn step_row_avx2_chunk(
        current: &[u8],
        row_base: usize,
        stride: usize,
        j: usize, // starting column (1..=cols) for this 32-byte chunk
        out_row: &mut [u8],
    ) {
        use std::arch::x86_64::*;
        // identical inner code as in step_kernel_avx2 for a single 32-byte chunk
        // compute idx
        let idx = row_base + j;
        let p_up = current.as_ptr().add(idx - stride);
        let p_cur = current.as_ptr().add(idx);
        let p_down = current.as_ptr().add(idx + stride);

        let lo_up = _mm_loadu_si128(p_up as *const __m128i);
        let hi_up = _mm_loadu_si128(p_up.add(16) as *const __m128i);
        let lo_cur = _mm_loadu_si128(p_cur as *const __m128i);
        let hi_cur = _mm_loadu_si128(p_cur.add(16) as *const __m128i);
        let lo_down = _mm_loadu_si128(p_down as *const __m128i);
        let hi_down = _mm_loadu_si128(p_down.add(16) as *const __m128i);

        let lo_up_l = _mm_loadu_si128(p_up.sub(1) as *const __m128i);
        let hi_up_l = _mm_loadu_si128(p_up.sub(1).add(16) as *const __m128i);
        let lo_cur_l = _mm_loadu_si128(p_cur.sub(1) as *const __m128i);
        let hi_cur_l = _mm_loadu_si128(p_cur.sub(1).add(16) as *const __m128i);
        let lo_down_l = _mm_loadu_si128(p_down.sub(1) as *const __m128i);
        let hi_down_l = _mm_loadu_si128(p_down.sub(1).add(16) as *const __m128i);

        let lo_up_r = _mm_loadu_si128(p_up.add(1) as *const __m128i);
        let hi_up_r = _mm_loadu_si128(p_up.add(1).add(16) as *const __m128i);
        let lo_cur_r = _mm_loadu_si128(p_cur.add(1) as *const __m128i);
        let hi_cur_r = _mm_loadu_si128(p_cur.add(1).add(16) as *const __m128i);
        let lo_down_r = _mm_loadu_si128(p_down.add(1) as *const __m128i);
        let hi_down_r = _mm_loadu_si128(p_down.add(1).add(16) as *const __m128i);

        let up_lo_i16 = _mm256_cvtepu8_epi16(lo_up);
        let cur_lo_i16 = _mm256_cvtepu8_epi16(lo_cur);
        let down_lo_i16 = _mm256_cvtepu8_epi16(lo_down);
        let up_l_lo_i16 = _mm256_cvtepu8_epi16(lo_up_l);
        let cur_l_lo_i16 = _mm256_cvtepu8_epi16(lo_cur_l);
        let down_l_lo_i16 = _mm256_cvtepu8_epi16(lo_down_l);
        let up_r_lo_i16 = _mm256_cvtepu8_epi16(lo_up_r);
        let cur_r_lo_i16 = _mm256_cvtepu8_epi16(lo_cur_r);
        let down_r_lo_i16 = _mm256_cvtepu8_epi16(lo_down_r);

        let up_hi_i16 = _mm256_cvtepu8_epi16(hi_up);
        let cur_hi_i16 = _mm256_cvtepu8_epi16(hi_cur);
        let down_hi_i16 = _mm256_cvtepu8_epi16(hi_down);
        let up_l_hi_i16 = _mm256_cvtepu8_epi16(hi_up_l);
        let cur_l_hi_i16 = _mm256_cvtepu8_epi16(hi_cur_l);
        let down_l_hi_i16 = _mm256_cvtepu8_epi16(hi_down_l);
        let up_r_hi_i16 = _mm256_cvtepu8_epi16(hi_up_r);
        let cur_r_hi_i16 = _mm256_cvtepu8_epi16(hi_cur_r);
        let down_r_hi_i16 = _mm256_cvtepu8_epi16(hi_down_r);

        let mut sum_lo = _mm256_add_epi16(cur_l_lo_i16, cur_r_lo_i16);
        sum_lo = _mm256_add_epi16(sum_lo, up_l_lo_i16);
        sum_lo = _mm256_add_epi16(sum_lo, up_lo_i16);
        sum_lo = _mm256_add_epi16(sum_lo, up_r_lo_i16);
        sum_lo = _mm256_add_epi16(sum_lo, down_l_lo_i16);
        sum_lo = _mm256_add_epi16(sum_lo, down_lo_i16);
        sum_lo = _mm256_add_epi16(sum_lo, down_r_lo_i16);

        let mut sum_hi = _mm256_add_epi16(cur_l_hi_i16, cur_r_hi_i16);
        sum_hi = _mm256_add_epi16(sum_hi, up_l_hi_i16);
        sum_hi = _mm256_add_epi16(sum_hi, up_hi_i16);
        sum_hi = _mm256_add_epi16(sum_hi, up_r_hi_i16);
        sum_hi = _mm256_add_epi16(sum_hi, down_l_hi_i16);
        sum_hi = _mm256_add_epi16(sum_hi, down_hi_i16);
        sum_hi = _mm256_add_epi16(sum_hi, down_r_hi_i16);

        let three = _mm256_set1_epi16(3);
        let mask_eq3_lo = _mm256_cmpeq_epi16(sum_lo, three);
        let mask_eq3_hi = _mm256_cmpeq_epi16(sum_hi, three);

        let two = _mm256_set1_epi16(2);
        let mask_eq2_lo = _mm256_cmpeq_epi16(sum_lo, two);
        let mask_eq2_hi = _mm256_cmpeq_epi16(sum_hi, two);

        let one = _mm256_set1_epi16(1);
        let alive_is_1_lo = _mm256_cmpeq_epi16(cur_lo_i16, one);
        let alive_is_1_hi = _mm256_cmpeq_epi16(cur_hi_i16, one);

        let tmp_lo = _mm256_and_si256(mask_eq2_lo, alive_is_1_lo);
        let survivors_lo = _mm256_or_si256(mask_eq3_lo, tmp_lo);

        let tmp_hi = _mm256_and_si256(mask_eq2_hi, alive_is_1_hi);
        let survivors_hi = _mm256_or_si256(mask_eq3_hi, tmp_hi);

        let packed = _mm256_packus_epi16(survivors_lo, survivors_hi);
        let out_lo_128 = _mm256_castsi256_si128(packed);
        let out_hi_128 = _mm256_extracti128_si256(packed, 1);

        use std::arch::x86_64::_mm_storeu_si128;
        _mm_storeu_si128(out_row.as_mut_ptr().add(j) as *mut _ as *mut _, out_lo_128);
        _mm_storeu_si128(
            out_row.as_mut_ptr().add(j + 16) as *mut _ as *mut _,
            out_hi_128,
        );
    }

    ///////////////////////////////////////////////////////////////////////////////
    // NEON kernel (aarch64)
    ///////////////////////////////////////////////////////////////////////////////
    #[cfg(all(target_arch = "aarch64"))]
    #[target_feature(enable = "neon")]
    unsafe fn step_kernel_neon(current: &[u8], rows: usize, cols: usize, out: &mut [u8]) {
        use std::arch::aarch64::*;
        let stride = cols + 2;
        out[stride..(rows + 1) * stride]
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(i0, out_row)| {
                let i = i0 + 1;
                let row_base = i * stride;
                // clear borders
                *out_row.get_unchecked_mut(0) = 0;
                *out_row.get_unchecked_mut(cols + 1) = 0;

                let mut j = 1usize;
                while j + 15 <= cols {
                    let idx = row_base + j;
                    let p_up = current.as_ptr().add(idx - stride);
                    let p_cur = current.as_ptr().add(idx);
                    let p_down = current.as_ptr().add(idx + stride);

                    let up = vld1q_u8(p_up);
                    let cur = vld1q_u8(p_cur);
                    let down = vld1q_u8(p_down);

                    let up_l = vld1q_u8(p_up.sub(1));
                    let cur_l = vld1q_u8(p_cur.sub(1));
                    let down_l = vld1q_u8(p_down.sub(1));

                    let up_r = vld1q_u8(p_up.add(1));
                    let cur_r = vld1q_u8(p_cur.add(1));
                    let down_r = vld1q_u8(p_down.add(1));

                    // widen to u16
                    let up_lo = vmovl_u8(vget_low_u8(up));
                    let up_hi = vmovl_u8(vget_high_u8(up));
                    let cur_lo = vmovl_u8(vget_low_u8(cur));
                    let cur_hi = vmovl_u8(vget_high_u8(cur));
                    let down_lo = vmovl_u8(vget_low_u8(down));
                    let down_hi = vmovl_u8(vget_high_u8(down));

                    let up_l_lo = vmovl_u8(vget_low_u8(up_l));
                    let up_l_hi = vmovl_u8(vget_high_u8(up_l));
                    let cur_l_lo = vmovl_u8(vget_low_u8(cur_l));
                    let cur_l_hi = vmovl_u8(vget_high_u8(cur_l));
                    let down_l_lo = vmovl_u8(vget_low_u8(down_l));
                    let down_l_hi = vmovl_u8(vget_high_u8(down_l));

                    let up_r_lo = vmovl_u8(vget_low_u8(up_r));
                    let up_r_hi = vmovl_u8(vget_high_u8(up_r));
                    let cur_r_lo = vmovl_u8(vget_low_u8(cur_r));
                    let cur_r_hi = vmovl_u8(vget_high_u8(cur_r));
                    let down_r_lo = vmovl_u8(vget_low_u8(down_r));
                    let down_r_hi = vmovl_u8(vget_high_u8(down_r));

                    let mut sum_lo = vaddq_u16(cur_l_lo, cur_r_lo);
                    sum_lo = vaddq_u16(sum_lo, up_l_lo);
                    sum_lo = vaddq_u16(sum_lo, up_lo);
                    sum_lo = vaddq_u16(sum_lo, up_r_lo);
                    sum_lo = vaddq_u16(sum_lo, down_l_lo);
                    sum_lo = vaddq_u16(sum_lo, down_lo);
                    sum_lo = vaddq_u16(sum_lo, down_r_lo);

                    let mut sum_hi = vaddq_u16(cur_l_hi, cur_r_hi);
                    sum_hi = vaddq_u16(sum_hi, up_l_hi);
                    sum_hi = vaddq_u16(sum_hi, up_hi);
                    sum_hi = vaddq_u16(sum_hi, up_r_hi);
                    sum_hi = vaddq_u16(sum_hi, down_l_hi);
                    sum_hi = vaddq_u16(sum_hi, down_hi);
                    sum_hi = vaddq_u16(sum_hi, down_r_hi);

                    let eq3_lo = vceqq_u16(sum_lo, vdupq_n_u16(3));
                    let eq2_lo = vceqq_u16(sum_lo, vdupq_n_u16(2));
                    let eq3_hi = vceqq_u16(sum_hi, vdupq_n_u16(3));
                    let eq2_hi = vceqq_u16(sum_hi, vdupq_n_u16(2));

                    let alive_lo = vceqq_u16(cur_lo, vdupq_n_u16(1));
                    let alive_hi = vceqq_u16(cur_hi, vdupq_n_u16(1));

                    let tmp_lo = vandq_u16(eq2_lo, alive_lo);
                    let surv_lo = vorrq_u16(eq3_lo, tmp_lo);

                    let tmp_hi = vandq_u16(eq2_hi, alive_hi);
                    let surv_hi = vorrq_u16(eq3_hi, tmp_hi);

                    let out_vec = vcombine_u8(vmovn_u16(surv_lo), vmovn_u16(surv_hi));
                    vst1q_u8(out_row.as_mut_ptr().add(j), out_vec);

                    j += 16;
                }

                // scalar tail
                let mut idx = row_base + j;
                while j <= cols {
                    let alive = *current.get_unchecked(idx);
                    let sum = *current.get_unchecked(idx - 1) as u16
                        + *current.get_unchecked(idx + 1) as u16
                        + *current.get_unchecked(idx - stride - 1) as u16
                        + *current.get_unchecked(idx - stride) as u16
                        + *current.get_unchecked(idx - stride + 1) as u16
                        + *current.get_unchecked(idx + stride - 1) as u16
                        + *current.get_unchecked(idx + stride) as u16
                        + *current.get_unchecked(idx + stride + 1) as u16;
                    *out_row.get_unchecked_mut(j) = ((sum == 3) || (alive == 1 && sum == 2)) as u8;
                    j += 1;
                    idx += 1;
                }
            });
    }

    ///////////////////////////////////////////////////////////////////////////////
    // End of file
    ///////////////////////////////////////////////////////////////////////////////

    pub fn gol_faster_stream() {
        let (rows, cols) = (X as usize, Y as usize);
        let mut grid = create_grid(rows, cols);
        let mut next = create_grid(rows, cols);

        // Random init into inner region
        let random_grid_timer: Instant = Instant::now();
        let mut rng = rand::rng();
        {
            let stride = cols + 2;
            for y in 0..rows {
                let base = (y + 1) * stride + 1;
                let row_slice = &mut grid[base..base + cols];
                rng.fill(row_slice);
                for v in row_slice.iter_mut() {
                    *v &= 1;
                }
            }
        }
        let random_grid_elapsed: u128 = random_grid_timer.elapsed().as_nanos();
        println!(
            "Random grid initialization time for a {}x{} grid: {} ns",
            X, Y, random_grid_elapsed
        );

        // Spawn ffmpeg to consume raw frames via stdin
        use std::io::Write;
        use std::process::{Command, Stdio};
        let output_name = format!("gol_simulation_fps_{}_X_{}_Y_{}.mp4", FPS, X, Y);
        let mut child = Command::new("ffmpeg")
            .args(&[
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-s",
                &format!("{}x{}", cols, rows),
                "-r",
                &format!("{}", FPS),
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                &output_name,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start ffmpeg. Is it installed and on PATH?");

        let mut ff_in = child.stdin.take().expect("Failed to open ffmpeg stdin");
        let mut frame_buf = vec![0u8; rows * cols];

        let mut sim_ns: u128 = 0;
        let mut io_ns: u128 = 0;
        for _iteration in 0..MAX_ITERATIONS {
            let timer_0 = Instant::now();
            //run the simulation step in parallel
            step_par(&grid, rows, cols, &mut next);
            sim_ns += timer_0.elapsed().as_nanos();

            // Swap so we save the just-computed generation from `grid`
            std::mem::swap(&mut grid, &mut next);

            let t1 = Instant::now();
            // Convert inner region to grayscale bytes (0/255)
            let stride = cols + 2;
            for y in 0..rows {
                let src_base = (y + 1) * stride + 1;
                let dst_base = y * cols;
                for x in 0..cols {
                    frame_buf[dst_base + x] = grid[src_base + x] * 255u8;
                }
            }
            ff_in
                .write_all(&frame_buf)
                .expect("Failed to write frame to ffmpeg");
            io_ns += t1.elapsed().as_nanos();
        }
        // Close stdin to signal EOF this ffmpeg to start encoding
        drop(ff_in);

        let status = child.wait().expect("Failed to wait on ffmpeg");

        //clear the console
        // print!("{}[2J", 27 as char);
        println!("gol_faster_stream()");
        println!("  Information:");
        println!("          Grid size: {}x{}", X, Y);
        println!("          Total iterations: {}", MAX_ITERATIONS);
        println!("  Summary:");
        println!("          Init time: {} ns", random_grid_elapsed);
        println!("          Simulation time: {} ns", sim_ns);
        println!("          I/O time (pipe to ffmpeg): {} ns", io_ns);
        println!(
            "Total time ms: {}",
            (random_grid_elapsed + sim_ns + io_ns) / 1_000_000
        );
        if status.success() {
            println!("Video created successfully: {}", output_name);
        } else {
            eprintln!("ffmpeg exited with status: {}", status);
        }
    }
}

fn setup() {
    // remove ./src/frames directory if it exists
    if std::path::Path::new("./src/frames").exists() {
        std::fs::remove_dir_all("./src/frames").unwrap();
    }
    // create ./src/frames directory
    std::fs::create_dir("./src/frames").unwrap();
}
// helper function to run ffmpeg to create a video from the generated PNG frames
fn run_ffmpeg(fps: u32, x: u32, y: u32) {
    use std::process::Command;

    let output = Command::new("ffmpeg")
        .args(&[
            "-framerate",
            format!("{}", fps).as_str(),
            "-i",
            "./src/frames/gol_%d.png",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            format!("gol_simulation_fps_{}_X_{}_Y_{}.mp4", fps, x, y).as_str(),
        ])
        .output()
        .expect("Failed to execute ffmpeg command");

    if output.status.success() {
        println!(
            "Video created successfully: gol_simulation_fps_{}_X_{}_Y_{}.mp4",
            fps, x, y
        );
    } else {
        eprintln!(
            "Error creating video: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

// main function
fn main() {
    // Use streaming pipeline to avoid slow per-frame PNG I/O
    setup();
    faster_hw::gol_faster_stream();
}
