Remove-Item -Path .\gol_simulation_fps_*_X_*_Y_*_M_*.mp4 -ErrorAction SilentlyContinue

cargo build --release
./target/release/gol_rs.exe faster_hw

# remove old mp4 files

# Find the latest output video (assuming naming convention)
$latest = Get-ChildItem -Path ./gol_simulation_fps_*_X_*_Y_*_M_*.mp4 | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latest) {
	$av1name = $latest.BaseName + "_av1.mp4"
	# Try AMD AMF AV1 hardware encoder first
	$amfTest = ffmpeg -hide_banner -encoders | Select-String "av1_amf"
	if ($amfTest) {
		ffmpeg -y -i $latest.FullName -c:v av1_amf -quality quality -usage transcoding -b:v 0 $av1name
		Write-Host "Re-encoded to AV1 (AMF hardware): $av1name"
	} else {
		ffmpeg -y -i $latest.FullName -c:v libaom-av1 -crf 30 -b:v 0 $av1name
		Write-Host "Re-encoded to AV1 (libaom): $av1name"
	}
} else {
	Write-Host "No output video found to re-encode."
}
# Remove all non-AV1 mp4 files (those not ending with _av1.mp4)
Get-ChildItem -Path .\gol_simulation_fps_*_X_*_Y_*_M_*.mp4 | Where-Object { $_.Name -notlike '*_av1.mp4' } | Remove-Item -ErrorAction SilentlyContinue
