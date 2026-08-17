#!/bin/bash

######## Config variables ---

TOTAL_DURATION=40 # test total duration in seconds 54000
SHIFT_DURATION=10             # shift duration in seconds 600
VIDEO_DURATION=3              # video duration in seconds 20
PROBLEM_DETECTED=false         # variable set to true if the MyRIO detects an error
GPIO_LINE=27 			       # numéro du GPIO BCM utilisé pour détecter l'erreur
GPIO_CHIP="gpiochip0" 


######## functions ---
# --- video recording function --- 

capture_video() {
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    FILENAME="$TEST_DIR/video_shift$(printf "%02d" $SHIFT_ID)_${TIMESTAMP}.h264"

    echo "Saving video : $FILENAME"

    libcamera-vid -o "$FILENAME" \
        -t $((VIDEO_DURATION * 1000)) \
        --framerate 42 \
        --width 1640 \
        --height 1232
}

# --- problem detection function ---
check_MyRIO(){
        echo "checking MyRIO status"
        VALUE=$(gpioget $GPIO_CHIP $GPIO_LINE)
        if [ "$VALUE" -eq 0 ]; then
            echo "problem detected"
            PROBLEM_DETECTED=true
        else
            echo "no problem detected"
        fi
}

######## main script
# --- Creating Test Folder ---

BASE_DIR="videos"
mkdir -p "$BASE_DIR"

LAST_TEST=$(ls -d "$BASE_DIR"/test_* 2>/dev/null | sed 's/.*test_//' | sort -n | tail -1)

if [ -z "$LAST_TEST" ]; then
    NEXT_TEST=1
else
    NEXT_TEST=$((LAST_TEST + 1))
fi

TEST_DIR="$BASE_DIR/test_$NEXT_TEST"
mkdir -p "$TEST_DIR"

echo "New folder created : $TEST_DIR"
echo "Videos will be saved here."


START=$SECONDS
END_TIME=$((START + TOTAL_DURATION))
SHIFT_ID=1

raspi-gpio set $GPIO_LINE pu # connect a pull-up to the pin connected to the MyRIO
# faut ajouter de set le 2e pin en high, pour lancer l'enregistrement du MyRIO

# --- Main Loop ---
while (( SECONDS < END_TIME )); do
    echo "=== Starting Shift $SHIFT_ID ==="

    SHIFT_START=$SECONDS
    SHIFT_END=$((SHIFT_START + SHIFT_DURATION))

    while (( SECONDS + VIDEO_DURATION < SHIFT_END )); do
        capture_video
        if [ "$PROBLEM_DETECTED" = false ]; then
            check_MyRIO
        else
            echo "a problem was detected in a previous video of this shift"
        fi
        
    done
    
    # problem detected means keep all the videos of this shift
    if [ "$PROBLEM_DETECTED" = true ]; then
        echo "a problem was detected in this shift, the videos have been kept"
    else
        echo "erasing unecessary videos of shift $SHIFT_ID"
        VIDEO_LIST=($(ls -t "$TEST_DIR"/video_shift$(printf "%02d" $SHIFT_ID)_*.h264)) # récupérer les noms de fichiers du plus récent au plus ancien
        if [ ${#VIDEO_LIST[@]} -gt 1 ]; then
            for video in "${VIDEO_LIST[@]:1}"; do
                echo "video $video deleted "
                rm "$video"
            done
        fi
    fi
    
    SHIFT_ID=$((SHIFT_ID + 1))
    PROBLEM_DETECTED=false

    while (( SECONDS < SHIFT_END )); do
        echo "Not enough time for a new recording, waiting for the end of the shift $SHIFT_ID"
        sleep 1
    done
done

echo "=== Video recording ended ==="
