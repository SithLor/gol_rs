to use 

rustup install nightly



cargo build --release
.\target\release\gol_rs.exe clear_results
.\target\release\gol_rs.exe all 100


to optizme build time run 
cargo clean 
cargo build --release --timings