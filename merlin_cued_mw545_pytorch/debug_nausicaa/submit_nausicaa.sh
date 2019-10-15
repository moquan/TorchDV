mv run_log_*.log log/ 2>/dev/null
./run_nausicaa.sh > run_log_"$(date +"%Y_%m_%d_%H_%M_%S").log" 2>&1   &
