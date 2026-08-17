"""
============================================================================================================
README
Ce fichier est actuellement la dernière version de ce code
Ce code sert à tuner les plages HSV des différentes couleurs pour la détection des marqueurs de couleur.

UTILISATION:
    -
TODO:
- faire marcher range_angles_histories et les plots associés
    a: on analyse par vidéo, et là on perds la référence angle_min entre les vidéos
    b: on analyse tout ensemble à la fin, mais là on a aucune idée de la durée de la vidéo (on pourrait le communiquer avec count...)
        ce qui fait  qu'on sait pas facilement retirer les pics parasites liés aux shifts
- rejeter les pics et vallées qui ne respectent pas une prominence suffisement proche de la moyenne avec leurs deux voisins
- pouvoir compiler en une run les vidéos dans plusieurs dossiers (ex: test106 et test107)
- ajouter le cas des couleurs rouges dans le fallback default frame

Sources ayant servi de base pour ce code:
- https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
- https://stackoverflow.com/questions/44588279/find-and-draw-the-largest-contour-in-opencv-on-a-specific-color-python
- https://stackoverflow.com/questions/28327020/opencv-detect-mouse-position-clicking-over-a-picture
- https://stackoverflow.com/questions/3426108/how-to-sort-a-list-of-strings-numerically
============================================================================================================
"""

#
# ===== Imports =====############################################################################################################
#
from AllVideoFunctions import * # import toutes les fonctions partagées entre les différents codes

from os import listdir
from os.path import isfile, join
import time
import re
from multiprocessing import Pool
from functools import partial




#
# ===== variables =====############################################################################################################
#
 
# ── Settings ────────────────────────────────────────────────────────
testName = "RunBaseLowAngle"
# pour analyser une partie du dossier, index de la première vidéo à traiter dans le dossier. 
start = 0               # defaut à 0 pour traiter toutes les vidéos
# pour analyser une partie du dossier, index suivant la dernière vidéo à traiter dans le dossier.
finish = 50             # defaut à -1 pour traiter toutes les vidéos
max_frames = 600        # nombre maximum de frames qu'on veut regarder par vidéo 
Clim = False             # True si on veut analyser les vidéos en climatisation, False sinon
tpu = False              # True si on veut analyser les vidéos avec du TPU, False sinon
FPS = 30                        # framerate des vidéos, nécessaire pour convertir les index de frames en temps en secondes
INTERVAL_BETWEEN_SHIFTS = 600   # secondes entre chaque shift

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

# ── Other variables ────────────────────────────────────────────────────────
frame_number = 0                    # numéro de la frame à lire (0 = première)
frame_display  = None
DISPLAY_WIDTH    = 1000              # largeur d'affichage en pixels
scale          = 1.0


# ======== functions ======############################################################################################################

def sort_human(l):
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split(r'([-+]?[0-9]*\.?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l



def frame_to_time(frame_index, shift_number, fps=FPS, interval=INTERVAL_BETWEEN_SHIFTS):
    """
    Converts a frame index to a time in seconds since the start of the recording.
    """
    return (shift_number) * interval + frame_index / fps

#attempt at multi-threading the video processing, which is the bottleneck of the code, to speed it up. Not working yet, need to check how to share the centroids_history variable between threads and how to avoid conflicts when writing to it at the same time from different threads.
def get_video_centroids_histories(video_tuple, folder_size, HSV_ranges, fingers_to_track, kernel, max_frames):
    l, video_path = video_tuple
    
    # if we ever need the actual shift number instead of 'l' 
    match = re.search(r'shift(\d+)', video_path)
    shift_number = int(match.group(1)) if match else -1

    # initialize an array to store the history of centroids positions for each color for each finger for this video
    local_centroids_histories = [[[] for _ in range(len(HSV_ranges))] for _ in range(len(fingers_to_track))]

    print(f"    Processing video {l+1}/{folder_size}...")
    # open the video
    cap = cv2.VideoCapture(video_path)
    
    # use the first frame to:
    #   - get the dimensions of the video
    #   - define the regions of interest 
    ret, frame = cap.read()
    width = frame.shape[1]
    height = frame.shape[0]

    # define the (x,y) locations of the 6 fingers in the frame, which must include the markers at ALL TIMES during the video
    # (0,0) is at the top left, x increases to the right, y increases to the bottom
    default_ROI_width = int(width*0.2)
    default_ROI_height = int(height*0.2)
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
    ROI_width = [[default_ROI_width]*len(HSV_ranges) for _ in range(6)]
    ROI_height = [[default_ROI_height]*len(HSV_ranges) for _ in range(6)]
    corners = [[tuple(default_corners[i]) for _ in range(len(HSV_ranges))] for i in range(6)]
    centroids = [[None]*len(HSV_ranges) for _ in range(6)]

    # set the sizes of ROI for each marker
    if len(HSV_ranges) == 4:
        focused_ROI_width = [100, 100, 170 , 270]
        focused_ROI_height = [100, 100, 130, 160]
    else:
        focused_ROI_width = [default_ROI_width]*len(HSV_ranges)
        focused_ROI_height = [default_ROI_height]*len(HSV_ranges)

    count = 1
    first_iteration = True

    # loop through the video frames
    while count <= max_frames: 

        # debug line to check which frame is being processed
        # print(f"Processing frame {count}...") 

        ret, frame = cap.read()
        if not ret:
            print(f"    End of video reached for video {l+1}/{folder_size} after {count} frames")
            break   


        for m,j in enumerate(fingers_to_track): # iterate through the fingers to track
            k = j-1
            for i in range(len(HSV_ranges)): # iterate through the colors to track

                # if we have a h_min bigger than h_max, it most likely means we are tracking red markers
                # (it means the color hue range goes aroung the 180 value back to zero)
                if HSV_ranges[i][0][0] > HSV_ranges[i][1][0]: 

                    # we need to combine two masks, one up to 180, and one from zero
                    frame_hsv = cv2.cvtColor(frame[corners[k][i][1]:corners[k][i][1]+ROI_height[k][i], corners[k][i][0]:corners[k][i][0]+ROI_width[k][i]], cv2.COLOR_BGR2HSV)
                    lower1 = np.array([0, HSV_ranges[i][0][1], HSV_ranges[i][0][2]])
                    upper1 = np.array([HSV_ranges[i][1][0], HSV_ranges[i][1][1], HSV_ranges[i][1][2]])
                    lower2 = np.array([HSV_ranges[i][0][0], HSV_ranges[i][0][1], HSV_ranges[i][0][2]])
                    upper2 = np.array([179, HSV_ranges[i][1][1], HSV_ranges[i][1][2]])
                    mask1 = cv2.inRange(frame_hsv, lower1, upper1)
                    mask2 = cv2.inRange(frame_hsv, lower2, upper2)
                    mask = cv2.bitwise_or(mask1, mask2)
                else:
                    mask = cv2.inRange(cv2.cvtColor(frame[corners[k][i][1]:corners[k][i][1]+ROI_height[k][i], corners[k][i][0]:corners[k][i][0]+ROI_width[k][i]], cv2.COLOR_BGR2HSV), HSV_ranges[i][0], HSV_ranges[i][1])

                # clean the mask by removing small contours and filling small holes
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                # keep only the largest contour in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                    
                    # add the centroid position to the history for this marker
                    local_centroids_histories[m][i].append((centroids[k][i][0] + corners[k][i][0], centroids[k][i][1] + corners[k][i][1]))

                    # update the corner of the region of interest for this color for the next frame, to have it centered on the centroid. max() is used to avoid getting out of the frame
                    corners[k][i] = (max(0, corners[k][i][0] + centroids[k][i][0] - ROI_width[k][i]//2), max(0, corners[k][i][1] + centroids[k][i][1] - ROI_height[k][i]//2))
                    
                else:# if the centroid is not detected in this frame, we search for it in the default frame
                    
                    # debug line to check when we need to use the full frame for the processing
                    print(f" Video {l+1}, Frame {count}, finger {j}, marker {i}: centroid not detected, using default frame for processing") 
                    default_frame = frame[default_corners[k][1]:default_corners[k][1] + default_ROI_height, default_corners[k][0]:default_corners[k][0]+default_ROI_width]
                    default_hsv = cv2.cvtColor(default_frame, cv2.COLOR_BGR2HSV)

                    if HSV_ranges[i][0][0] > HSV_ranges[i][1][0]: 
                    
                        # we need to combine two masks, one up to 180, and one from zero
                        lower1 = np.array([0, HSV_ranges[i][0][1], HSV_ranges[i][0][2]])
                        upper1 = np.array([HSV_ranges[i][1][0], HSV_ranges[i][1][1], HSV_ranges[i][1][2]])
                        lower2 = np.array([HSV_ranges[i][0][0], HSV_ranges[i][0][1], HSV_ranges[i][0][2]])
                        upper2 = np.array([179, HSV_ranges[i][1][1], HSV_ranges[i][1][2]])
                        mask1 = cv2.inRange(default_hsv, lower1, upper1)
                        mask2 = cv2.inRange(default_hsv, lower2, upper2)
                        mask = cv2.bitwise_or(mask1, mask2)
                    else:
                        mask = cv2.inRange(default_hsv, HSV_ranges[i][0], HSV_ranges[i][1]) 
                    
                    # get the centroids of the masked areas for each color
                    centroids[k][i] = get_centroid(mask)
                    
                    if centroids[k][i][0] is None or centroids[k][i][1] is None: # if the centroid is detected in the default frame, we flip it back to the original orientation of the frame, to be able to add it to the full frame later
                        print(f" TUNE YOUR RANGES!!! Video {l+1},Frame {count}, finger {j}, marker {i}: centroid not detected in default frame of interest") 
                        print(f" either check your HSV ranges for marker {i} or check the size and position of the default frame of interest for this finger") # debug line to check when we need to use the previous position for the processing
                        local_centroids_histories[m][i].append(centroids[k][i])
                        continue
    
                    local_centroids_histories[m][i].append((centroids[k][i][0] + default_corners[k][0], centroids[k][i][1] + default_corners[k][1]))
                                   
                    corners[k][i] = (max(0, default_corners[k][0] + centroids[k][i][0] - ROI_width[k][i]//2), max(0, default_corners[k][1] + centroids[k][i][1] - ROI_height[k][i]//2))
        if first_iteration:
            first_iteration = False
            # get angle offest of each phalanx for each finger, based on the first frame of the video
            '''if shift_number == 1:  # only get the angle offsets for the first video, as they should be the same for all videos
                angleOffsets = getAngleOffsets(frame, local_centroids_histories, fingers_to_track)'''
                
        count += 1

    cap.release()
    cv2.destroyAllWindows()

    ## ======== computing angles ========

    # compute the angles based on the history of centroids positions
    local_angles_histories = getAnglesHistories(fingers_to_track, local_centroids_histories,l+1)
    local_peak_angles_histories = get_peak_angles_histories(fingers_to_track, local_angles_histories)
    local_valley_angles_histories = get_valley_angles_histories(fingers_to_track, local_angles_histories)
    local_range_angles_histories = get_range_angles_histories(fingers_to_track, local_peak_angles_histories, local_valley_angles_histories)
    
    # remplacer les frame_index par des temps en secondes dans les histories
    for m in range(len(fingers_to_track)):
        
        # angles histories : [q1_angles, q2_angles, frame_indexes]
        q1_angles, q2_angles, frame_indexes = local_angles_histories[m]
        local_angles_histories[m] = [q1_angles, q2_angles, [frame_to_time(f, shift_number) for f in frame_indexes]]

        q1_peak_angles, q1_peak_frames, q2_peak_angles, q2_peak_frames = local_peak_angles_histories[m]
        local_peak_angles_histories[m] = [q1_peak_angles, [frame_to_time(f, shift_number) for f in q1_peak_frames], q2_peak_angles, [frame_to_time(f, shift_number) for f in q2_peak_frames]]

        q1_valley_angles, q1_valley_frames, q2_valley_angles, q2_valley_frames = local_valley_angles_histories[m]
        local_valley_angles_histories[m] = [q1_valley_angles, [frame_to_time(f, shift_number) for f in q1_valley_frames], q2_valley_angles, [frame_to_time(f, shift_number) for f in q2_valley_frames]]

        q1_range_vals, q1_range_frames, q2_range_vals, q2_range_frames = local_range_angles_histories[m]
        local_range_angles_histories[m] = [q1_range_vals, [frame_to_time(f, shift_number) for f in q1_range_frames], q2_range_vals, [frame_to_time(f, shift_number) for f in q2_range_frames]]
    

    return local_centroids_histories, local_angles_histories, local_peak_angles_histories, local_valley_angles_histories, local_range_angles_histories, count-1
    
############################################################################################################
#
## ======== Main ========
#
############################################################################################################

if __name__ == "__main__":
    time_start = time.perf_counter()

    ## ======== Analysing videos ========

    # Getting filepath and markers colors
    Folder_path = Folder_paths[testName]
    Markers_colors = TestMarkers[testName]

    # define the HSV ranges for each color
    HSV_ranges = [(color_ranges[Markers_colors[i]][0], color_ranges[Markers_colors[i]][1]) for i in range(len(Markers_colors))]

    # initialize an array to store the history of centroids positions for each color for each finger
    centroids_histories = [[[] for _ in range(len(Markers_colors))] for _ in range(len(fingers_to_track))]
    angles_histories = [[[] for _ in range(3)] for _ in range(len(fingers_to_track))]
    peak_angles_histories = [[[] for _ in range(4)] for _ in range(len(fingers_to_track))]
    valley_angles_histories = [[[] for _ in range(4)] for _ in range(len(fingers_to_track))]
    range_angles_histories = [[[] for _ in range(4)] for _ in range(len(fingers_to_track))]

    # open and sort the video folder
    videos_paths = [join(Folder_path, f) for f in listdir(Folder_path) if isfile(join(Folder_path, f)) and f.endswith(".h264")]
    sorted_videos_paths = sort_human(videos_paths)[start:finish]
    folder_size = len(sorted_videos_paths)

    # others variables initialization
    width = -1
    height = -1
    total_frames = 0
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    window_name    = "Point selector"
    current_group  = 0
    current_points = []
    all_selections = []
    POINTS_PER_GROUP = 3                 

    GROUP_COLORS = [        # Couleur affichées par groupe (BGR)
        (255,   0,   0),    # bleu
        (0,   140, 255),    # orange
        (0,   200,   0),    # vert
        (0,   0,   255),    # rouge
        (255,   0, 200),    # violet
        (0,   200, 255),    # jaune, a remplacer par brun pour match les plots
    ]

    #loop through the entire folder
    print(f"=== Found {folder_size} videos in the folder {Folder_path} ===")
    print(f"Processing the videos...")


    indexed_videos = list(enumerate(sorted_videos_paths))  # [(0, path0), (1, path1), ...]

    # créer une fonction avec les arguments fixes
    process_video = partial(
        get_video_centroids_histories,
        folder_size=folder_size,
        HSV_ranges=HSV_ranges,
        fingers_to_track=fingers_to_track,
        kernel=kernel,
        max_frames=max_frames
    )

    with Pool(processes=None) as pool: # None = utilise tous les cœurs disponibles
        results = pool.map(process_video, indexed_videos)

    for video_centroids_histories,video_angles_histories, video_peak_angles_histories, video_valley_angles_histories,video_range_angles_histories, video_frame_count in results:
        for m in range(len(fingers_to_track)):
            for i in range(len(HSV_ranges)):
                centroids_histories[m][i].extend(video_centroids_histories[m][i])
            for i in range(3):
                angles_histories[m][i].extend(video_angles_histories[m][i])
            for i in range(4):
                peak_angles_histories[m][i].extend(video_peak_angles_histories[m][i])
                valley_angles_histories[m][i].extend(video_valley_angles_histories[m][i])
                range_angles_histories[m][i].extend(video_range_angles_histories[m][i])
        total_frames += video_frame_count
        

    end_time = time.perf_counter()
    print(f"Time taken to process {len(sorted_videos_paths)} videos: {end_time - time_start} seconds")
    print(f"Total frames processed: {total_frames}")
    
    ## ======== Plotting results ========

    print(f"Plotting results...")
    plot_angle_evolution(fingers_to_track, angles_histories)
    plot_peak_angles_evolution(fingers_to_track, peak_angles_histories)
    plot_valley_angles_evolution(fingers_to_track, valley_angles_histories)
    plot_range_histories(fingers_to_track, range_angles_histories)
    #plot_markers_histories(centroids_histories, fingers_to_track, Markers_colors,width, height)

    plt.show()

