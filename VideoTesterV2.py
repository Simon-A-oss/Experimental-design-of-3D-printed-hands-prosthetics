"""
============================================================================================================
READMEEEEEE
Ce fichier est un fichier pour debug l'analyse vidéo.
Il sert d'étape intermédiaire entre ColorRangesTuning.py et VideoFolderAnalysisV2.py,
pour vérifier que l'analyse vidéo fonctionne correctement avant d'en extraire des données.
Pour que ce code fonctionne correctement, il faut d'abord régler les plages HSV des différents marqueurs 
à l'aide du code ColorRangesTuning.py
Il permet de:
    - visualiser les zones d'intéret par défaut pour chaque doigt et les zones de chaque marqueur de chaque doigt
    - visualiser les masques créés pour chaque marqueur de chaque doigt
    - visualiser les positions des centroids détectés pour chaque marqueur de chaque doigt
    - vérifier les plages angulaires détectées pour chaque marqueur de chaque doigt sur la durée de la vidéo
UTILISATION:
    -
TODO:
    - régler le bug de display quand found_centroids[k][i] = False (quand on utilise la default_frame[k][i] )
    -> les coordonnees où est mis le centroid sont pas bonnes

Sources ayant servi de base pour ce code:
-
============================================================================================================
"""

from AllVideoFunctions import *


# ===== variables =====############################################################################################################

# ── Settings ────────────────────────────────────────────────────────
testName = "RunBlancs"
video_path = video_paths[testName] # change this to test a specific video

max_frames = 900       # nombre de frames qu'on veut regarder, à régler en fonction de la durée de la vidéo et de la puissance de calcul de l'ordinateur, pour éviter les temps de traitement trop longs
Clim = False             # True si on veut analyser les vidéos en climatisation, False sinon
tpu = False              # True si on veut analyser les vidéos avec du TPU, False sinon         
fingers_to_track = [1,2,3,4,5,6]  # List of fingers to track, going from 1 to 6
# fingers positions in the frame:
###########################
#                 4       #
#       3                 #
#                 5       #
#                         #
#       2                 #
#                 6       #
#       1                 #
#                         #
###########################


# ======== functions ======############################################################################################################




# ======== Main ========############################################################################################################

# define the HSV ranges for each color
Markers_colors = TestMarkers[testName]
HSV_ranges = [(color_ranges[Markers_colors[i]][0], color_ranges[Markers_colors[i]][1]) for i in range(len(Markers_colors))]

# initialize an array to store the history of centroids positions for each color for each finger
centroids_history = [[[] for _ in range(len(Markers_colors))] for _ in range(len(fingers_to_track))]


# open the video file
cap = cv2.VideoCapture(video_path)

# use the first frame to:
#   - get the dimensions of the video
#   - define the regions of interest
ret, frame = cap.read()
width = frame.shape[1]
height = frame.shape[0]

# initialize an array to store the history of centroids positions for each color for each finger

default_ROI_width = int(width*0.2)
default_ROI_height = int(height*0.2)
ROI_width = [[default_ROI_width]*len(Markers_colors) for _ in range(6)]
ROI_height = [[default_ROI_height]*len(Markers_colors) for _ in range(6)]

# define the (x,y) locations of the 6 fingers in the frame, which must include the markers at ALL TIMES during the video
# (0,0) is at the top left, x increases to the right, y increases to the bottom
if Clim:
    climOffset_x = 60
    climOffset_y = 30
else:
    climOffset_x = 0
    climOffset_y = 0
if tpu:
    tpuOffset_x = 20
    tpuOffset_y = 30
else:
    tpuOffset_x = 0
    tpuOffset_y = 0

default_corners = [(int(width*0.05)+climOffset_x, int(height*0.8)-climOffset_y - tpuOffset_y),     # top left corner of zone of interest for finger 1 (bottom left of the frame)
                   (int(width*0.05)+climOffset_x, int(height*0.47)-climOffset_y + tpuOffset_y),    # top left corner of zone of interest for finger 2 (middle left of the frame)
                   (int(width*0.05)+climOffset_x, int(height*0.1)-climOffset_y + tpuOffset_y),     # top left corner of zone of interest for finger 3 (top left of the frame)
                   (int(width*0.73)+climOffset_x+tpuOffset_x, int(height*0.03)-climOffset_y + tpuOffset_y),    # top left corner of zone of interest for finger 4 (top right of the frame)
                   (int(width*0.73)+climOffset_x+tpuOffset_x, int(height*0.35)-climOffset_y - tpuOffset_y),    # top left corner of zone of interest for finger 5 (middle right of the frame)
                   (int(width*0.73)+climOffset_x+tpuOffset_x, int(height*0.74)-climOffset_y - tpuOffset_y)]    # top left corner of zone of interest for finger 6 (bottom right of the frame)

default_ROI = [(default_corners[i][0], default_corners[i][1], default_corners[i][0] + default_ROI_width, default_corners[i][1] + default_ROI_height) for i in range(6)]

# initialize the zone of interest for each color, for each finger
# these will be updated later in the code based on the detected centroids positions
corners = [[tuple(default_corners[i]) for _ in range(len(Markers_colors))] for i in range(6)]
previous_corners = [[tuple(default_corners[i]) for _ in range(len(Markers_colors))] for i in range(6)]
regions_of_interest = [[None for _ in range(len(Markers_colors))] for _ in range(6)]
frames_hsv = [[None for _ in range(len(Markers_colors))] for _ in range(6)]
masks = [[None for _ in range(len(Markers_colors))] for _ in range(6)]
centroids = [[None]*len(Markers_colors) for _ in range(6)]

previous_corners = [[tuple(c) for c in row] for row in corners]

# set the sizes of ROI for each marker.
if len(Markers_colors) == 5:
    focused_ROI_width = [40, 120, 120, 210 , 270]
    focused_ROI_height = [40, 120, 120, 130, 160]
elif len(Markers_colors) == 4:
    focused_ROI_width = [120, 120, 210 , 270]
    focused_ROI_height = [120, 120, 130, 160]
else:
    focused_ROI_width = [default_ROI_width]*len(Markers_colors)
    focused_ROI_height = [default_ROI_height]*len(Markers_colors)

first_iteration = True
found_centroids = [[False]*len(Markers_colors) for _ in range(6)] 


count = 1
wait_time = 0  # time to wait between frames in milliseconds, can be adjusted for faster or slower processing
still_testing = True

while count <= max_frames: 

    # debug line to check which frame is being processed
    # print(f"Processing frame {count}...") 

    ret, frame = cap.read()
    if not ret:
        print("End of video reached")
        break   

    for m,j in enumerate(fingers_to_track): # iterate through the fingers to track
    
        k = j-1
        for i in range(len(HSV_ranges)): # iterate through the colors to track
            #while True:
            # debug line to check the position and size of the frame of interest for each color, for each finger
            #print(f"Frame {count}, finger {k}, color {i}, window corner: {corners[k][i]}, size: ({ROI_width[k][i]}, {ROI_height[k][i]})") 

            # reduce the search region to accelerate processing
            regions_of_interest[k][i] = frame[corners[k][i][1]:corners[k][i][1]+ROI_height[k][i], 
                                            corners[k][i][0]:corners[k][i][0]+ROI_width[k][i]]
            
            # update previous corners for the full_frame_display_all(...) function
            previous_corners[k][i] = corners[k][i]
            
            # convert the frame of interest to HSV color space for color detection
            frames_hsv[k][i] = cv2.cvtColor(regions_of_interest[k][i], cv2.COLOR_BGR2HSV) 

            # create the masks for each color based on the defined HSV ranges
            #if the range goes beyond 180° in hue, we need to create two masks
            if HSV_ranges[i][0][0] > HSV_ranges[i][1][0]:
                lower1 = np.array([0, HSV_ranges[i][0][1], HSV_ranges[i][0][2]])
                upper1 = np.array([HSV_ranges[i][1][0], HSV_ranges[i][1][1], HSV_ranges[i][1][2]])
                lower2 = np.array([HSV_ranges[i][0][0], HSV_ranges[i][0][1], HSV_ranges[i][0][2]])
                upper2 = np.array([179, HSV_ranges[i][1][1], HSV_ranges[i][1][2]])
                mask1 = cv2.inRange(frames_hsv[k][i], lower1, upper1)
                mask2 = cv2.inRange(frames_hsv[k][i], lower2, upper2)
                masks[k][i] = cv2.bitwise_or(mask1, mask2)
            else:
                masks[k][i] = cv2.inRange(frames_hsv[k][i], HSV_ranges[i][0], HSV_ranges[i][1])

            # clean the mask by removing small contours and filling small holes
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            masks[k][i] = cv2.morphologyEx(masks[k][i], cv2.MORPH_OPEN, kernel)
            masks[k][i] = cv2.morphologyEx(masks[k][i], cv2.MORPH_CLOSE, kernel)

            # keep only the largest contour in the mask
            contours, _ = cv2.findContours(masks[k][i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                biggestBlob = None
            else:
                biggestBlob = max(contours, key = cv2.contourArea)


            # get the centroids of the masked areas for each color
            centroids[k][i] = get_centroid(biggestBlob)
            if first_iteration: 
                ROI_width[k][i] = focused_ROI_width[i]
                ROI_height[k][i] = focused_ROI_height[i]
            
            if centroids[k][i][0] is not None and centroids[k][i][1] is not None: # if the centroid is detected, we add it to the history and we update the position of the frame of interest for the next frame
                found_centroids[k][i] = True
                # debug line to check the HSV values around the centroid positions
                #check_area_HSV_values(frames_of_interest[k][i], frames_hsv[k][i], masks[k][i], j, i, centroids[k][i][0]-9, centroids[k][i][1], size=10) 
                
                # add the centroid position to the history for this marker
                centroids_history[m][i].append((centroids[k][i][0] + corners[k][i][0], centroids[k][i][1] + corners[k][i][1]))

                # update the corner of the region of interest for this color for the next frame, to have it centered on the centroid. max() is used to avoid getting out of the frame
                corners[k][i] = (max(0, corners[k][i][0] + centroids[k][i][0] - ROI_width[k][i]//2), max(0, corners[k][i][1] + centroids[k][i][1] - ROI_height[k][i]//2))
                
                

            else:# if the centroid is not detected in this frame, we search for it in the default frame
                found_centroids[k][i] = False

                # debug line to check when we need to use the full frame for the processing
                print(f"Frame {count}, finger {j}, marker {i}: centroid not detected, using default frame for processing") 
                
                default_frame = frame[default_corners[k][1]:default_corners[k][1] + default_ROI_height, default_corners[k][0]:default_corners[k][0]+default_ROI_width]
                
                default_hsv = cv2.cvtColor(default_frame, cv2.COLOR_BGR2HSV)
                #if the range goes beyond 180° in hue, we need to create two masks
                if HSV_ranges[i][0][0] > HSV_ranges[i][1][0]:
                    lower1 = np.array([0, HSV_ranges[i][0][1], HSV_ranges[i][0][2]])
                    upper1 = np.array([HSV_ranges[i][1][0], HSV_ranges[i][1][1], HSV_ranges[i][1][2]])
                    lower2 = np.array([HSV_ranges[i][0][0], HSV_ranges[i][0][1], HSV_ranges[i][0][2]])
                    upper2 = np.array([179, HSV_ranges[i][1][1], HSV_ranges[i][1][2]])
                    mask1 = cv2.inRange(default_hsv, lower1, upper1)
                    mask2 = cv2.inRange(default_hsv, lower2, upper2)
                    masks[k][i] = cv2.bitwise_or(mask1, mask2)
                else:
                    masks[k][i] = cv2.inRange(default_hsv, HSV_ranges[i][0], HSV_ranges[i][1])
                
                # get the centroids of the masked areas for each color
                centroids[k][i] = get_centroid(masks[k][i])
                
                if centroids[k][i][0] is None or centroids[k][i][1] is None: # if the centroid is detected in the default frame, we flip it back to the original orientation of the frame, to be able to add it to the full frame later
                    print(f" TUNE YOUR RANGES!!! Frame {count}, finger {j}, marker {i}: centroid not detected in default frame of interest") 
                    print(f" either check your HSV ranges for color {i} or check the position of the default frame of interest for this finger") 
                    centroids_history[m][i].append(centroids[k][i])
                    continue

                # debug line to check the centroids positions in the default frame
                print(f"    Centroid for Frame {count}, finger {j}, marker {i} detected in default frame of interest: {centroids[k][i]}")
                centroids_history[m][i].append((centroids[k][i][0] + default_corners[k][0], centroids[k][i][1] + default_corners[k][1])) 
                corners[k][i] = (max(0, default_corners[k][0] + centroids[k][i][0] - ROI_width[k][i]//2), max(0, default_corners[k][1] + centroids[k][i][1] - ROI_height[k][i]//2))
                
        # uncomment to wait for a key press after processing one finger
        #cv2.waitKey(0)

    copy_frame = frame.copy()
    full_frame_display_everything(copy_frame, default_ROI, masks, HSV_ranges, fingers_to_track, previous_corners, ROI_width, ROI_height, centroids, found_centroids, first_iteration, count, show_centroids=True, show_ROIs=True, show_masks=True)
    
    if first_iteration:
        first_iteration = False

        # get angle offest of each phalanx for each finger, based on the first frame of the video
        #angleOffsets = getAngleOffsets(frame, centroids_history, fingers_to_track)

    count += 1
    # wait for user input
    key = cv2.waitKey(wait_time)  # wait for a key press after processing a full frame
    if key == 32:  # space key
        continue  # continue to the next frame
    elif key == 27:  # escape key
        print("Testing stopped by user.")
        still_testing = False
        break  # exit the loop and stop the testing

cap.release()
cv2.destroyAllWindows()

if still_testing:

    ## ======== computing angles ========
    print(f"Computing angles evolution...")
    # compute the angles based on the history of centroids positions
    angles_histories = getAnglesHistories(fingers_to_track, centroids_history)
    peak_angles_histories = get_peak_angles_histories(fingers_to_track, angles_histories)
    valley_angles_histories = get_valley_angles_histories(fingers_to_track, angles_histories)
    range_angles_histories = get_range_angles_histories(fingers_to_track, peak_angles_histories, valley_angles_histories)

    ## ======== Plotting results ========
    print(f"Plotting results...")
    plot_angle_evolution(fingers_to_track, angles_histories)
    plot_peak_angles_evolution(fingers_to_track, peak_angles_histories)
    plot_valley_angles_evolution(fingers_to_track, valley_angles_histories)
    plot_range_histories(fingers_to_track, range_angles_histories)
    plot_markers_histories(centroids_history, fingers_to_track, Markers_colors,width, height)

    plt.show()
