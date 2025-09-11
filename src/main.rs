const CONFIG_FPS: u32 = 60;// 24, 30, 60 
const CONFIG_VIDEO_LENGTH_SECONDS: u32 = 10;// 10, 30, 60
const CONFIG_X: u32 = 500; // 1920, 2560, 3840 ,
const CONFIG_Y: u32 = 500; // 1080, 1440, 2160 
const CONFIG_SAVE_LOCATION: &str = "./src/frames";
const CONFIG_MAX_ITERATIONS: u32 = CONFIG_FPS * CONFIG_VIDEO_LENGTH_SECONDS;

use std::path;
use std::sync::RwLock;
use std::time::Instant;

use rayon::str::SplitWhitespace;
static METHOD: RwLock<&str> = RwLock::new(" ");


mod faster {
    // 512x512 take 0,32 sec so 1920

    const FPS: u32 = super::CONFIG_FPS;
    const VIDEO_LENGTH_SECONDS: u32 = super::CONFIG_VIDEO_LENGTH_SECONDS;
    const MAX_ITERATIONS: u32 = super::CONFIG_FPS * super::CONFIG_VIDEO_LENGTH_SECONDS;
    const X: u32 = super::CONFIG_X; // 4k resolution
    const Y: u32 = super::CONFIG_Y; // 4k resolution
    const SAVE_LOCATION: &str = super::CONFIG_SAVE_LOCATION;

    use rand::Rng;
    use rayon::prelude::*;
    use std::io::Write;
    use std::process::{Command, Stdio};
    use std::time::Instant;

    use crate::append_results_to_file;

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
    fn gol_faster_stream() {
        *super::METHOD.write().unwrap() = "faster_hw_agnostic";

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
        let t = *super::METHOD.read().unwrap();
        //println!("{}()", t);
        let output_name = format!("gol_simulation_fps_{}_X_{}_Y_{}_M_{}.mp4", FPS, X, Y, t);

      


        let output_name = format!("gol_simulation_fps_{}_X_{}_Y_{}_M_{}.mp4", FPS, X, Y, t);
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
                "libx265",
                "-preset",
                "medium",
                "-pix_fmt",
                "gray",
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

        let status: std::process::ExitStatus = child.wait().expect("Failed to wait on ffmpeg");

        //clear the console
        //print!("{}[2J", 27 as char);
        append_results_to_file(X, Y, MAX_ITERATIONS, random_grid_elapsed, sim_ns, io_ns);
        println!("{}()", *super::METHOD.read().unwrap());
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
            //println!("Video created successfully: {}", output_name);
        } else {
            eprintln!("ffmpeg exited with status: {}", status);
        }
    }
    pub fn RUN() {
        gol_faster_stream();
    }
}

mod faster_hw {
    // 512x512 take 0,32 sec so 1920

    pub const FPS: u32 = super::CONFIG_FPS;
    pub const VIDEO_LENGTH_SECONDS: u32 = super::CONFIG_VIDEO_LENGTH_SECONDS;

    pub const MAX_ITERATIONS: u32 = super::CONFIG_FPS * super::CONFIG_VIDEO_LENGTH_SECONDS;
    pub const X: u32 = super::CONFIG_X; // 4k resolution
    pub const Y: u32 = super::CONFIG_Y; // 4k resolution
    pub const SAVE_LOCATION: &str = super::CONFIG_SAVE_LOCATION;

    use crate::METHOD;
    use rand::Rng;
    use rayon::prelude::*;

    use std::sync::RwLock;
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
    pub fn step_par(current: &[u8], rows: usize, cols: usize, out: &mut [u8], mode: &RwLock<&str>) {
        let stride = cols + 2;
        debug_assert_eq!(current.len(), (rows + 2) * stride);
        debug_assert_eq!(out.len(), (rows + 2) * stride);

        // Zero top and bottom padded rows
        out[..stride].fill(0);
        out[(rows + 1) * stride..].fill(0);

        // Dispatch to the best kernel available at runtime
        unsafe {
            step_kernel_simd_dispatch(current, rows, cols, out, mode);
        }
    }

    /// Runtime dispatch to AVX-512 (if available), AVX2, NEON, or scalar fallback.
    /// This is unsafe because kernels use unchecked loads/stores.
    #[inline(always)]
    unsafe fn step_kernel_simd_dispatch(
        current: &[u8],
        rows: usize,
        cols: usize,
        out: &mut [u8],
        mode: &RwLock<&str>,
    ) {


       
        step_kernel_neon(current, rows, cols, out);
    

        *mode.write().unwrap() = "fast_hw_scalar";
        unsafe {
            step_kernel_scalar(current, rows, cols, out);
        }
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
                unsafe {
                    *out_row.get_unchecked_mut(0) = 0;
                    *out_row.get_unchecked_mut(cols + 1) = 0;
                }

                let mut idx = row_base + 1;
                for j in 1..=cols {
                    unsafe {
                        let alive = *current.get_unchecked(idx);
                        let sum = *current.get_unchecked(idx - 1) as u16
                            + *current.get_unchecked(idx + 1) as u16
                            + *current.get_unchecked(idx - stride - 1) as u16
                            + *current.get_unchecked(idx - stride) as u16
                            + *current.get_unchecked(idx - stride + 1) as u16
                            + *current.get_unchecked(idx + stride - 1) as u16
                            + *current.get_unchecked(idx + stride) as u16
                            + *current.get_unchecked(idx + stride + 1) as u16;
                        *out_row.get_unchecked_mut(j) =
                            ((sum == 3) || (alive == 1 && sum == 2)) as u8;
                    }
                    idx += 1;
                }
            });
    }

 
    ///////////////////////////////////////////////////////////////////////////////
    // NEON kernel (aarch64)
    ///////////////////////////////////////////////////////////////////////////////
 
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

    pub fn gol_faster_stream() {
        //rogh fix beace METHOD is not defined in this scope in step_par->step_kernel_avx512_>process_block_avx512
        static MODE: &RwLock<&str> = &METHOD;

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
        //println!(
        //    "Random grid initialization time for a {}x{} grid: {} ns",
        //    X, Y, random_grid_elapsed
        //);

        // Spawn ffmpeg to consume raw frames via stdin
        use std::io::Write;
        use std::process::{Command, Stdio};
        // Run one simulation step to set the method name
        step_par(&grid, rows, cols, &mut next, &MODE);
        std::mem::swap(&mut grid, &mut next);
        let method_name = *MODE.read().unwrap();
        let output_name = format!(
            "gol_simulation_fps_{}_X_{}_Y_{}_M_{}.mp4",
            FPS, X, Y, method_name
        );

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
                "medium",
                "-pix_fmt",
                "gray",
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
        // Already did first step above, so start from 1
        for _iteration in 1..MAX_ITERATIONS {
            let timer_0 = Instant::now();
            step_par(&grid, rows, cols, &mut next, &MODE);
            sim_ns += timer_0.elapsed().as_nanos();
            std::mem::swap(&mut grid, &mut next);

            let t1 = Instant::now();
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

        //save data
        super::append_results_to_file(X, Y, MAX_ITERATIONS, random_grid_elapsed, sim_ns, io_ns);

        //append method name to mp4 file name before extension

        println!("{}()", *super::METHOD.read().unwrap());
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
            //println!("Video created successfully: {}", output_name);
        } else {
            eprintln!("ffmpeg exited with status: {}", status);
        }
    }
    pub fn RUN() {
        gol_faster_stream();
    }
}

fn append_results_to_file(
    x: u32,
    y: u32,
    iterations: u32,
    init_ns: u128,
    sim_ns: u128,
    io_ns: u128,
) {
    use std::fs::OpenOptions;
    use std::io::Write;

    let file_path = "results.csv";
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_path)
        .expect("Unable to open or create results.csv");
    writeln!(
        file,
        "{},{},{},{},{},{},{},{}",
        *METHOD.read().unwrap(),
        x,
        y,
        iterations,
        init_ns,
        sim_ns,
        io_ns,
        (init_ns + sim_ns + io_ns)
    )
    .expect("Unable to write data to results.txt");
}
fn setup_results_file() {
    use std::fs::OpenOptions;
    use std::io::Write;
    let file_path = "results.csv";
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(file_path)
        .expect("Unable to open or create results.csv");
    writeln!(file, "Method,X,Y,Iterations,Init_ns,Sim_ns,IO_ns,Total_ns")
        .expect("Unable to write header to results.txt");
}

fn flush_cache() {
    let cache_size = 38 * 1024 * 1024; // 64MB, adjust as needed for your CPU
    let mut buffer = vec![0u8; cache_size];
    for i in 0..cache_size {
        buffer[i] = i as u8;
    }
    std::hint::black_box(&buffer);
}

fn main() {
    //check if results file exists, if not create it and add header
    if !std::path::Path::new("results.csv").exists() {
        setup_results_file();
    }

    let valid_args = [
        "faster",
        "faster_hw",
        "all",
        "multi_faster",
        "multi_faster_hw",
        "clear_results",
    ];
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || !valid_args.contains(&args[1].as_str()) {
        eprintln!("Usage: {} <method> <how_many_runs>", args[0]);
        eprintln!("  method: one of {:?}", valid_args);
        std::process::exit(1);
    }
    //take user input for how many runs
    let runs: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    println!("if you want to change config edit top of src/main.rs and recompile the program");
    //wait
    // std::thread::sleep(std::time::Duration::from_secs(2));
    print!("{}[2J", 27 as char);

    match args[1].as_str() {
        "all" => {
            for _ in 0..runs {
                flush_cache();
                faster::RUN();
            }
            for _ in 0..runs {
                flush_cache();
                faster_hw::RUN();
            }
        }
        "multi_stock" => {
            //alert the complie
            // slower::RUN();
            println!("The stock version is disabled in this build. it take too long to run.");
        }
        "multi_faster" => {
            for _ in 0..runs {
                flush_cache();
                faster::RUN();
            }
        }
        "multi_faster_hw" => {
            for _ in 0..runs {
                flush_cache();
                faster_hw::RUN();
            }
        }
        "faster" => {
            flush_cache();
            faster::RUN();
        }
        "faster_hw" => {
            flush_cache();
            faster_hw::RUN();
        }
        "clear_results" => {
            setup_results_file();
            println!("results.txt file cleared");
        }
        _ => {
            faster_hw::RUN();
        }
    }
}
