# remove ./src/frames directory if it exists
if (Test-Path -Path "./src/frames") {
    Remove-Item -Recurse -Force "./src/frames"
}
# create ./src/frames directory
New-Item -ItemType Directory -Path "./src/frames" | Out-Null
# run cargo run
cargo run -r 
ffmpeg -framerate 30 -i ./src/frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p gol_simulation.mp4