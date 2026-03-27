methods="PEARL"

domains="office" 
# domains="pinball" 
# domains="logistics"  
# domains="goal"

config="0"

partitioning="flexible"
# partitioning="uniform"

trials="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20" 
result_dir="results"

method_dir="$methods"_"$partitioning"_td_v
# method_dir="$methods"_"$partitioning"_td
# method_dir="$methods"_"$partitioning"_v


echo $method_dir
# Runs all the methods, domains, and trials on different threads

for method in $methods; do
    for domain in $domains; do
        for trial in $trials; do
            echo $method
            echo $domain
            echo $trial
            log_directory="$result_dir"/"$domain"/"$method_dir"/trial_"$trial"
            log_path=$log_directory/"$trial".log
            echo $log_path

            if [ ! -d "$log_directory" ]
            then
                echo "File doesn't exist. Creating now"
                mkdir -p ./$log_directory
                echo "File created"
            else
                echo "File exists"
            fi
            
            if [ "$method" = PEARL ]; then
            python3 main.py --yaml yamls/"$domain"_"$partitioning"_"$config".yaml  --config "$config" --trial "$trial" --result_dir "$result_dir" --method_dir "$method_dir" > "$log_path" 2>&1 &
            echo "Trial $trial started and logging to $log_path"
            fi 

        done
    done
done