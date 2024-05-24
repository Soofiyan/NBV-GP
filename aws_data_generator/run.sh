#!/bin/bash

# Path to the Python script
PYTHON_SCRIPT="suction_deformation_score_calculator_multi_angle_center_view.py"

# Log file
LOG_FILE="run_log.txt"

# Function to check if all tasks are completed
check_completion() {
    # Implement your logic to check if all tasks are done
    # For example, check for the existence of a completion flag/file
    # Return 0 if tasks are complete, 1 otherwise

    # Example check (replace with your actual condition)
    if [ -f "all_tasks_done.flag" ]; then
        return 0
    else
        return 1
    fi
}

# Run the script continuously until all tasks are done
while true; do
    echo "Running Python script..." | tee -a $LOG_FILE
    python $PYTHON_SCRIPT >> $LOG_FILE 2>&1

    # Check the exit status of the Python script
    if [ $? -eq 0 ]; then
        # Python script executed successfully
        if check_completion; then
            echo "All tasks completed successfully." | tee -a $LOG_FILE
            break
        else
            echo "Tasks not complete. Retrying..." | tee -a $LOG_FILE
        fi
    else
        # Python script encountered an error
        echo "Python script encountered an error. Retrying..." | tee -a $LOG_FILE
    fi

    sleep 1  # Optional: add a delay before restarting
done

echo "Script finished." | tee -a $LOG_FILE
