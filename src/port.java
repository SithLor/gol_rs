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

    let file_path = "results_java.csv";
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
    let file_path = "results_java.csv";
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(file_path)
        .expect("Unable to open or create results.csv");
    writeln!(file, "Method,X,Y,Iterations,Init_ns,Sim_ns,IO_ns,Total_ns")
        .expect("Unable to write header to results.txt");
}

fn main() {
    if !std::path::Path::new("results_java.csv").exists() {
        setup_results_file();
    }
    faster::RUN();
}