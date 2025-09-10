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
        println!("Total time ms: {}", (random_grid_elapsed + sim_ns + io_ns) / 1_000_000);
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
    faster::gol_faster_stream();
}
