#!/bin/bash
# WU-14 Run Monitor — checks Modal app status every 90 minutes
# Writes to scripts/monitor_log.txt

LOG="/Users/charlie/all/misalign-fv.wu-14/scripts/monitor_log.txt"
INTERVAL=5400  # 90 minutes

# Our Modal app IDs
APP_IDS="ap-bI275YzTPBS3TOrtfURM0i ap-OPk2L5lXUWr603ei8cEBWy ap-xkBobQFSIc53ZoGZikEleK ap-Wz2hsFCm4vhbe0EkW3yGrT ap-G2YwZonHJreybZcfsickAd ap-twEZeAIBVg0HHb0kAgrOxO"
APP_NAMES="fv_inverted/42 fv_inverted/123 fv_inverted/456 ut_inverted random_reward zero_reward"

check_status() {
    echo "========================================" >> "$LOG"
    echo "CHECK @ $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$LOG"
    echo "========================================" >> "$LOG"

    # Dump full modal app list (just our apps)
    echo "" >> "$LOG"
    echo "Modal app status:" >> "$LOG"

    ids_arr=($APP_IDS)
    names_arr=($APP_NAMES)
    all_stopped=true

    for i in "${!ids_arr[@]}"; do
        app_id="${ids_arr[$i]}"
        name="${names_arr[$i]}"
        # Use modal app list and grep for this specific app
        line=$(modal app list 2>/dev/null | grep "$app_id")
        if [ -n "$line" ]; then
            echo "  $name: $line" >> "$LOG"
            if echo "$line" | grep -q "ephemeral"; then
                all_stopped=false
            fi
        else
            echo "  $name ($app_id): NOT IN LIST (may have completed and been cleaned up)" >> "$LOG"
        fi
    done

    # Check checkpoints
    echo "" >> "$LOG"
    echo "Checkpoints on Modal volume:" >> "$LOG"
    for dir in fv_inverted ut_inverted random_reward zero_reward; do
        contents=$(modal volume ls misalign-checkpoints "/checkpoints/$dir/" 2>/dev/null)
        if [ -n "$contents" ]; then
            echo "  /checkpoints/$dir/:" >> "$LOG"
            echo "$contents" | while read line; do
                echo "    $line" >> "$LOG"
            done
        else
            echo "  /checkpoints/$dir/: (empty or not yet created)" >> "$LOG"
        fi
    done

    echo "" >> "$LOG"

    if $all_stopped; then
        echo "*** ALL RUNS APPEAR COMPLETE ***" >> "$LOG"
        return 1
    fi
    return 0
}

echo "WU-14 Run Monitor started at $(date -u '+%Y-%m-%d %H:%M:%S UTC')" > "$LOG"
echo "Checking every $((INTERVAL/60)) minutes" >> "$LOG"
echo "App IDs: $APP_IDS" >> "$LOG"
echo "" >> "$LOG"

# Initial check
check_status

# Loop until all done
while [ $? -eq 0 ]; do
    sleep $INTERVAL
    check_status
done

echo "Monitor exiting at $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$LOG"
