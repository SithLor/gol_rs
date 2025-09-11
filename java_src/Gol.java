import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.Random;
import java.util.concurrent.*;
import javax.imageio.ImageIO;
import java.util.stream.IntStream;

public class Gol {
    // Configuration constants
    static final int FPS = 30;
    static final int VIDEO_LENGTH_SECONDS = 10;
    static final int MAX_ITERATIONS = FPS * VIDEO_LENGTH_SECONDS;
    static final int X = 512; // grid width
    static final int Y = 512; // grid height
    static final String SAVE_LOCATION = ".";

    static String METHOD = "faster_java";

    public static void main(String[] args) throws Exception {
        if (!Files.exists(Paths.get("results_java.csv"))) {
            setupResultsFile();
        }
        run();
    }

    static void run() throws Exception {
        golFasterStream();
    }

    static void golFasterStream() throws Exception {
        int rows = X, cols = Y;
        byte[] grid = createGrid(rows, cols);
        byte[] next = createGrid(rows, cols);

        // Random init into inner region
        long randomGridStart = System.nanoTime();
        Random rng = new Random();
        int stride = cols + 2;
        for (int y = 0; y < rows; y++) {
            int base = (y + 1) * stride + 1;
            for (int x = 0; x < cols; x++) {
                grid[base + x] = (byte)(rng.nextInt(2));
            }
        }
        long randomGridElapsed = System.nanoTime() - randomGridStart;
        System.out.printf("Random grid initialization time for a %dx%d grid: %d ns%n", X, Y, randomGridElapsed);

        // ffmpeg process
        String outputName = String.format("gol_simulation_fps_%d_X_%d_Y_%d_M_%s.mp4", FPS, X, Y, METHOD);
        ProcessBuilder pb = new ProcessBuilder(
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", cols + "x" + rows, "-r", String.valueOf(FPS), "-i", "pipe:0",
            "-c:v", "libx265", "-preset", "medium", "-pix_fmt", "gray", outputName
        );
        pb.redirectOutput(ProcessBuilder.Redirect.DISCARD);
        pb.redirectError(ProcessBuilder.Redirect.PIPE);
        Process ffmpeg = pb.start();
        OutputStream ffIn = ffmpeg.getOutputStream();

        byte[] frameBuf = new byte[rows * cols];
        long simNs = 0, ioNs = 0;

        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
            long simStart = System.nanoTime();
            stepPar(grid, rows, cols, next);
            simNs += System.nanoTime() - simStart;

            // Swap
            byte[] tmp = grid; grid = next; next = tmp;

            long ioStart = System.nanoTime();
            // Convert inner region to grayscale bytes (0/255)
            for (int y = 0; y < rows; y++) {
                int srcBase = (y + 1) * stride + 1;
                int dstBase = y * cols;
                for (int x = 0; x < cols; x++) {
                    frameBuf[dstBase + x] = (byte)(grid[srcBase + x] * 255);
                }
            }
            ffIn.write(frameBuf);
            ioNs += System.nanoTime() - ioStart;
        }
        ffIn.close();
        int status = ffmpeg.waitFor();

        appendResultsToFile(X, Y, MAX_ITERATIONS, randomGridElapsed, simNs, ioNs);

        System.out.println(METHOD + "()");
        System.out.println("  Information:");
        System.out.printf("          Grid size: %dx%d%n", X, Y);
        System.out.printf("          Total iterations: %d%n", MAX_ITERATIONS);
        System.out.println("  Summary:");
        System.out.printf("          Init time: %d ns%n", randomGridElapsed);
        System.out.printf("          Simulation time: %d ns%n", simNs);
        System.out.printf("          I/O time (pipe to ffmpeg): %d ns%n", ioNs);
        System.out.printf("Total time ms: %d%n", (randomGridElapsed + simNs + ioNs) / 1_000_000);
        if (status == 0) {
            // Video created successfully
        } else {
            System.err.println("ffmpeg exited with status: " + status);
        }
    }

    static byte[] createGrid(int rows, int cols) {
        return new byte[(rows + 2) * (cols + 2)];
    }

    static void stepPar(byte[] current, int rows, int cols, byte[] out) throws InterruptedException, ExecutionException {
        int stride = cols + 2;
        // Zero top and bottom padded rows
        for (int i = 0; i < stride; i++) out[i] = 0;
        for (int i = (rows + 1) * stride; i < out.length; i++) out[i] = 0;

        ForkJoinPool pool = ForkJoinPool.commonPool();
        pool.submit(() -> IntStream.range(1, rows + 1).parallel().forEach(i -> {
            int rowBase = i * stride;
            out[rowBase] = 0;
            out[rowBase + cols + 1] = 0;
            for (int j = 1; j <= cols; j++) {
                int idx = rowBase + j;
                int alive = current[idx];
                int sum =
                    current[idx - 1] + current[idx + 1] +
                    current[idx - stride - 1] + current[idx - stride] + current[idx - stride + 1] +
                    current[idx + stride - 1] + current[idx + stride] + current[idx + stride + 1];
                out[idx] = (byte)(((sum == 3) || (alive == 1 && sum == 2)) ? 1 : 0);
            }
        })).get();
    }

    static void appendResultsToFile(int x, int y, int iterations, long initNs, long simNs, long ioNs) throws IOException {
        try (BufferedWriter file = new BufferedWriter(new FileWriter("results_java.csv", true))) {
            file.write(String.format("%s,%d,%d,%d,%d,%d,%d,%d%n",
                METHOD, x, y, iterations, initNs, simNs, ioNs, (initNs + simNs + ioNs)));
        }
    }

    static void setupResultsFile() throws IOException {
        try (BufferedWriter file = new BufferedWriter(new FileWriter("results_java.csv"))) {
            file.write("Method,X,Y,Iterations,Init_ns,Sim_ns,IO_ns,Total_ns\n");
        }
    }
}