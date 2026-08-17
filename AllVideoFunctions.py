"""
============================================================================================================
README

============================================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import find_peaks


#=================================================================================================================
#-----------------------------------------------------------------------------------------------------------------
#   !!!    -- Update 'color_ranges' after using the ColorRangeTuning.py code -- !!!
# global variables
#   !!!    -- Update 'color_ranges' after using the ColorRangeTuning.py code -- !!!
#-----------------------------------------------------------------------------------------------------------------
#=================================================================================================================

color_ranges  = {
    "red-orange": (np.array([0, 65, 169]), np.array([14, 255, 255])),
    "orange": (np.array([20, 140, 204]),np.array([32, 255, 255])),
    #"orange8000": (np.array([20, 51, 200]), np.array([31, 255, 255])),
    "orange2": (np.array([20, 147, 211]), np.array([31, 255, 255])),
    "orange8000": (np.array([22, 120, 120]), np.array([32, 255, 255])),
    "yellow": (np.array([27, 80, 200]),np.array([33, 255, 255])),
    "yellow_clim": (np.array([22, 80, 90]), np.array([32, 255, 255])),
    "yellow_slow": (np.array([22, 80, 90]), np.array([31, 255, 255])),
    #"light_green": (np.array([33, 127, 205]), np.array([44, 255, 255])),
    "light_green": (np.array([33, 92, 91]), np.array([50, 255, 255])),
    "light_green3": (np.array([33, 48, 83]), np.array([63, 255, 255])),
    "light_green_clim": (np.array([33, 92, 91]), np.array([58, 255, 255])),
    "light_green_slow": (np.array([33, 118, 90]), np.array([58, 255, 255])),
    "green": (np.array([45, 50, 225]),np.array([70, 255, 255])),
    "dark_green": (np.array([38, 35, 174]), np.array([74, 151, 231])),
    "dark_green": (np.array([48, 35, 108]), np.array([80, 252, 255])),
    "dark_green2": (np.array([54, 57, 81]), np.array([87, 160, 160])),
    "dark_green3": (np.array([64, 78, 50]), np.array([88, 255, 255])),
    "dark_green_clim": (np.array([56, 60, 91]), np.array([88, 160, 194])),
    "dark_green_clim2": (np.array([60, 60, 91]), np.array([88, 255, 255])),
    "dark_green_clim3": (np.array([56, 77, 69]), np.array([88, 160, 196])),
    "dark_green_slow": (np.array([54, 79, 52]), np.array([83, 160, 162])),             
    "light_blue": (np.array([95, 80, 102]), np.array([115, 255, 255])),
    "light_blue1": (np.array([96, 37, 180]), np.array([155, 255, 255])),
    "light_blue2": (np.array([89, 39, 196]), np.array([138, 255, 245])),
    "navy_blue": (np.array([120, 150, 85]),np.array([130, 255, 130])),
    "red-blueish": (np.array([170, 120, 120]),np.array([180, 255, 255])),
    "full-red": (np.array([170, 65, 107]), np.array([8, 255, 255])),
    "full-red3": (np.array([170, 91, 95]), np.array([8, 255, 255])),
    "full-red_clim": (np.array([170, 90, 125]), np.array([14, 255, 255])),
    "full-red_clim3": (np.array([170, 90, 98]), np.array([10, 255, 255])),
    "full-red_Slow": (np.array([170, 65, 74]), np.array([9, 255, 255])),
    "P1": (np.array([13, 80, 185]), np.array([29, 255, 255])),
    "P2": (np.array([44, 36, 66]), np.array([72, 255, 255])),
    "P3": (np.array([30, 72, 111]), np.array([43, 255, 255])),
    "P4": (np.array([90, 64, 64]), np.array([116, 255, 255])),

}

TestMarkers = {
    "RunBaseLowAngle" :         ["light_green", "light_blue2", "dark_green", "orange8000"],
    "RunBaseLowAngle2" :        ["light_green", "light_blue2", "dark_green", "orange8000"],
    "RunQuartSpeed" :           ["dark_green_slow", "light_green_slow", "yellow_slow", "full-red_Slow"],
    "RunBaseModel-Ultimaker" :  ["light_green", "dark_green", "orange2", "light_blue1"],
    "RunBlancs" :               ["dark_green", "red-orange", "light_blue2","orange8000"],
    "Runbleus" :                ["dark_green2", "light_green", "full-red","orange8000"],
    "Runtpu" :                  ["dark_green3", "light_green3", "full-red3","orange8000"],
    "RunClimBAseModel" :         ["dark_green_clim", "light_green", "yellow_clim", "full-red_clim"],
    "RunClimModded" :          ["dark_green_clim2", "light_green_clim", "yellow_clim", "full-red_clim"],
    "RunClimTPU" :             ["dark_green_clim3", "light_green_clim", "yellow_clim", "full-red_clim3"],
    "RunPauline" :             ["P1", "P2", "P3", "P4"],
}    

Folder_paths = {
    "RunBaseLowAngle" :         "Runs\\FatigueTests\\RunBaseLowAngle\\test_106\\",
    "RunBaseLowAngle2" :        "Runs\\FatigueTests\\RunBaseLowAngle\\test_107\\",
    "RunQuartSpeed" :           "Runs\\FatigueTests\\RunQuartSpeed\\test_151\\",
    "RunBaseModel-Ultimaker" :  "Runs\\FatigueTests\\RunBaseModel-Ultimaker\\test_103",
    "RunBlancs" :               "Runs\\FatigueTests\\RunBlancs\\test_113",
    "Runbleus" :                "Runs\\FatigueTests\\Runbleus\\test_124",
    "Runtpu" :                  "Runs\\FatigueTests\\Runtpu\\test_121",
    "RunClimBAseModel" :         "Runs\\FatigueTests\\RunClimBAseModel\\test_140",
    "RunClimModded" :          "Runs\\FatigueTests\\RunClimModded\\test_144",
    "RunClimTPU" :             "Runs\\FatigueTests\\RunClimTPU\\test_147"
}

video_paths = {
    "RunBaseLowAngle" :         "Runs\\FatigueTests\\RunBaseLowAngle\\test_106\\video_shift01_2026-03-06_17-39-04.h264",
    "RunBaseLowAngle2" :        "Runs\\FatigueTests\\RunBaseLowAngle\\test_107\\video_shift01_2026-03-10_17-39-01.h264",
    "RunQuartSpeed" :           "Runs\\FatigueTests\\RunQuartSpeed\\test_151\\video_shift01_2026-05-11_19-50-08.h264",
    "RunBaseModel-Ultimaker" :  "Runs\\FatigueTests\\RunBaseModel-Ultimaker\\test_103\\video_shift01_2026-02-27_16-17-20.h264",
    "RunBlancs" :               "Runs\\FatigueTests\\RunBlancs\\test_113\\video_shift01_2026-03-12_16-30-26.h264",
    "Runbleus" :                "Runs\\FatigueTests\\Runbleus\\test_124\\video_shift01_2026-03-25_12-42-43.h264",
    "Runtpu" :                  "Runs\\FatigueTests\\Runtpu\\test_121\\video_shift01_2026-03-24_16-09-12.h264",
    "RunClimBAseModel" :        "Runs\\FatigueTests\\RunClimBAseModel\\test_140\\video_shift01_2026-04-27_20-19-21.h264",
    "RunClimModded" :           "Runs\\FatigueTests\\RunClimModded\\test_144\\video_shift01_2026-04-29_15-46-19.h264",
    "RunClimTPU" :              "Runs\\FatigueTests\\RunClimTPU\\test_147\\video_shift01_2026-05-04_14-42-33.h264",
    "RunPauline" :              "demoVideos\\video_2024-07-18_13-55-36.h264"
}


kernel_size = 3
min_area = 30











#=================================================================================================================
#-----------------------------------------------------------------------------------------------------------------
#
# Functions used for markers detection and tracking (VideoTester.py and VideoFolderAnalysis.py)
#
#-----------------------------------------------------------------------------------------------------------------
#=================================================================================================================

NUM_GROUPS       = 6                 # nombre de groupes (= nombre de doigts)
POINTS_PER_GROUP = 3                 # points par groupe (3 points pour les articulations du doigt, 4 points pour les marqueurs)
DISPLAY_WIDTH    = 1000              # <-- largeur d'affichage en pixels (adapte à ton écran)
 
# Couleur par groupe (BGR)
GROUP_COLORS = [
    (0,   0,   255),  # rouge
    (0,   200,   0),  # vert
    (255,   0,   0),  # bleu
    (0,   200, 255),  # jaune
    (255,   0, 200),  # violet
    (0,   140, 255),  # orange
]
 
# ── Variables globales ────────────────────────────────────────────────────────
current_group  = 0
current_points = []
all_selections = []
frame_display  = None
scale          = 1.0
window_name    = "Point selector"
 
 # ── fonctions ────────────────────────────────────────────────────────

def draw_status():
    remaining = POINTS_PER_GROUP - len(current_points)
    color     = GROUP_COLORS[current_group] if current_group < NUM_GROUPS else (200, 200, 200)
    label     = (f"Group {current_group + 1}/{NUM_GROUPS}  -  "
                 f"click {remaining} more point(s)   |   'q' to quit")
    cv2.rectangle(frame_display, (0, 0), (frame_display.shape[1], 30), (30, 30, 30), -1)
    cv2.putText(frame_display, label, (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
 
 
def mouse_callback(event, x, y, flags, param):
    global current_group, current_points, all_selections, frame_display
 
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if current_group >= NUM_GROUPS:
        return
    if len(current_points) >= POINTS_PER_GROUP:
        return
 
    # Coordonnées réelles dans la frame originale
    real_x = int(x / scale)
    real_y = int(y / scale)
 
    current_points.append((real_x, real_y))
    color    = GROUP_COLORS[current_group]
    pt_index = len(current_points)
 
    # Dessine sur l'image affichée (coordonnées écran x, y)
    cv2.circle(frame_display, (x, y), 3, color, -1)
    #cv2.putText(frame_display, f"G{current_group + 1}P{pt_index}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
 
    # ← Rafraîchit immédiatement l'affichage pour que le dernier point soit visible
    cv2.imshow(window_name, frame_display)
    cv2.waitKey(1)
 
    print(f"  Group {current_group + 1} — P{pt_index} : x={real_x}, y={real_y}")
 
    # Groupe complet ?
    if len(current_points) == POINTS_PER_GROUP:
        selection = list(current_points)
        all_selections.append(selection)
        print(f"  → selection{current_group + 1} = {selection}")
 
        current_group  += 1
        current_points  = []
 
        if current_group < NUM_GROUPS:
            print(f"\nNow click {POINTS_PER_GROUP} points for group {current_group + 1}:\n")
            draw_status()
            cv2.imshow(window_name, frame_display)
            cv2.waitKey(1)
        else:
            print("\nAll groups selected! Press any key to close.")
 
 
def get_refCoords(frame, centroids_history):
    """
    Description:
        Opens a video file, reads a specific frame, and allows the user to click on points in the frame to select reference coordinates
    parameters:
        frame (ndarray): The frame to process.
        centroids_history (list): A list of lists of tuples containing the centroid x-y positions for each finger.
        
    returns:
        all_selections (list): A list of lists of tuples containing the selected points for each group.
    """
    global frame_display, scale
 
    # ── Rescale pour l'affichage ──────────────────────────────────────────────
    orig_h, orig_w = frame.shape[:2]
    scale          = DISPLAY_WIDTH / orig_w
    display_h      = int(orig_h * scale)

    # Afficher les centroids sur la frame
    for  finger_centroids in centroids_history:
        for history in finger_centroids:
            centroid_x, centroid_y = history[0]
            # Convertir les coordonnées du centroid en coordonnées d'affichage
            display_x = int(centroid_x * scale)
            display_y = int(centroid_y * scale)
            # Dessiner un cercle pour chaque centroid
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # Rouge pour les centroids
            
    frame_display  = cv2.resize(frame, (DISPLAY_WIDTH, display_h))
    print("\n")
    print(f"Frame loaded — original: {orig_w}x{orig_h} px  |  "
          f"displayed: {DISPLAY_WIDTH}x{display_h} px  (scale={scale:.3f})")
    print(f"Click {POINTS_PER_GROUP} points for each of the {NUM_GROUPS} groups.")
    print("Press 'q' to quit early.\n")
    print("── Group 1 ──────────────────────────────")
    print(f"Now click {POINTS_PER_GROUP} points for group 1:\n")
 
    draw_status()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
 
    while True:
        cv2.imshow(window_name, frame_display)
        key = cv2.waitKey(20) & 0xFF
 
        if key == ord('q'):
            print("Quit early.")
            break
        if current_group >= NUM_GROUPS:
            cv2.waitKey(0)
            break
 
    cv2.destroyAllWindows()

    return all_selections

def getAngleOffsets(frame, centroids_history, fingers_to_track):
    # 1) for each finger:
    #   1.1) get the coordinates of the centroids of each marker -> we already have them in centroids_history
    # 2) display a frame with the cendroids marked by a circle and the user can click on the articulations of each finger to get their coordinates
    # and 3) extract the coordinates of the articulations of each finger based on user input
    refCoords = get_refCoords(frame,centroids_history)
        # il faut encore modifier cette fonction pour qu'elle display les centroids sur la frame
    # 4) compute the angles of each phalanx based on the articulations coordinates
    articulationsAngles = []
    for id_doigt in range(len(fingers_to_track)):
        angleUn = getAngle2(refCoords[id_doigt][0][0],refCoords[id_doigt][0][1],refCoords[id_doigt][1][0],refCoords[id_doigt][1][1])
        orientedUn = orientAnglev5(angleUn,id_doigt+1,1)
        angleDeux = getAngle2(refCoords[id_doigt][1][0],refCoords[id_doigt][1][1],refCoords[id_doigt][2][0],refCoords[id_doigt][2][1])
        orientedDeux = orientAnglev5(angleDeux,id_doigt+1,2)
        articulationsAngles.append((orientedUn, orientedDeux))
    # 5) compute the offset between the centroid based angles and the articulation based angles for each phalanx of each finger
    # 6) store the offsets in a dictionary to be used later in the code (in the getAnglesHistories function)
    angleOffsets = {}
    for id_doigt in range(len(fingers_to_track)):
        markerAngle1 = getAngle2(centroids_history[id_doigt][0][0][0],centroids_history[id_doigt][0][0][1],centroids_history[id_doigt][1][0][0],centroids_history[id_doigt][1][0][1])
        markerAngle1o = orientAnglev5(markerAngle1,id_doigt+1,1)
        markerAngle2 = getAngle2(centroids_history[id_doigt][2][0][0],centroids_history[id_doigt][2][0][1],centroids_history[id_doigt][3][0][0],centroids_history[id_doigt][3][0][1])
        markerAngle2o = orientAnglev5(markerAngle2,id_doigt+1,2)
        deltaUn = ((markerAngle1o - articulationsAngles[id_doigt][0] + 180) % 360) - 180
        deltaDeux = ((markerAngle2o - articulationsAngles[id_doigt][1] + 180) % 360) - 180
        angleOffsets[fingers_to_track[id_doigt]] = (deltaUn, deltaDeux)
        print(f"Finger {fingers_to_track[id_doigt]}: angle offset for phalanx 1: {deltaUn:.2f}, angle offset for phalanx 2: {deltaDeux:.2f}")

#############




def get_centroid(mask):
    """
    Description:
        Calculate the centroid of the masked area.
    Parameters:
        mask (ndarray): A binary image where the area of interest is white (255) and the rest is black (0).
    Returns:
        centroid_x (int): The x-coordinate of the centroid.
        centroid_y (int): The y-coordinate of the centroid.
    """
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
        return centroid_x, centroid_y
    else:
        return None, None
    
def full_frame_display_everything(frame, default_frames_of_interest, masks, HSV_ranges, fingers_to_track, corners, ROI_width, ROI_height, centroids, found_centroids, first_iteration, count, show_masks=True, show_ROIs=True, show_centroids=True):
    """
    Description:
        debug function
        display the full frame with:
            - the default zone of interest for each finger in green if it is being tracked, in blue otherwise
            - the current zones of interest for each marker, for each finger, in green
            - the masked areas for each color, for each finger
    Parameters:
        frame (ndarray): The current frame being processed.
        default_frames_of_interest (list of list of int): The default frames of interest for each finger, in the format [(x1, y1, x2, y2), (x1, y1, x2, y2), ...]
        corners (list of list of list of int): The top left corners of the current frames of interest for each color, for each finger.
        ROI_width (list of list of int): The width of the current frames of interest for each color, for each finger.
        ROI_height (list of list of int): The height of the current frames of interest for each color, for each finger.
        centroids (list of list of tuple): The centroids positions for each color, for each finger
        found_centroids (list of list of bool): A list indicating whether the centroid was found for each color, for each finger
        first_iteration (bool): A boolean indicating whether this is the first iteration of the frame processing loop
        count (int): The frame count.
        show_masks (bool): A boolean indicating whether to show the masked areas for each color, for each finger, in the full frame.
        show_ROIs (bool): A boolean indicating whether to show the current zones of interest for each marker, for each finger, in the full frame.
        show_centroids (bool): A boolean indicating whether to show the centroids positions for each color, for each finger, in the full frame.
    """
    width = frame.shape[1]
    height = frame.shape[0]
    copied_frame = frame.copy()
    for a in range(len(corners)):
        
        if a+1 in fingers_to_track:
            # show the positions of the tracked fingers in the full frame
            cv2.rectangle(frame, (default_frames_of_interest[a][0], default_frames_of_interest[a][1]), (default_frames_of_interest[a][2], default_frames_of_interest[a][3]), (0, 150, 0), 4) # debug line to check the default frames of interest for each finger
            
            if show_ROIs:
                for b in range(len(corners[a])):
                    # test the positions of the 5 colors for each finger
                    cv2.rectangle(frame, (corners[a][b][0], corners[a][b][1]), (corners[a][b][0]+ROI_width[a][b], corners[a][b][1]+ROI_height[a][b]), (0, 255, 0), 2) # debug line to check the frame of interest used for the processing

            if show_masks:
                for b in range(len(corners[a])):
                    # add the masks in the frame
                    frame = insert_mask(frame, masks[a][b], corners[a][b][0], corners[a][b][1], default_frames_of_interest[a], found_centroids[a][b], first_iteration)
                
            if show_centroids:
                for b in range(len(corners[a])):
                    # add the centroids positions in the frame
                    if centroids[a][b][0] is not None and centroids[a][b][1] is not None:
                        # if centroid is found without using the default fram
                        if found_centroids[a][b]:
                            cv2.circle(frame, (centroids[a][b][0] + corners[a][b][0], centroids[a][b][1] + corners[a][b][1]), 5, (0, 0, 255), -1) # debug line to check the positions of the centroids in the full frame
                        else:
                            # if centroid is found using the default frame, we can display it in a different color to see if the default frame is well defined
                            cv2.circle(frame, (centroids[a][b][0] + default_frames_of_interest[a][0], centroids[a][b][1] + default_frames_of_interest[a][1]), 5, (255, 0, 0), -1) # debug line to check the positions of the centroids in the full frame
        else:
            
            # show the positions of the non tracked fingers in the full frame
            cv2.rectangle(frame, (default_frames_of_interest[a][0], default_frames_of_interest[a][1]), (default_frames_of_interest[a][2], default_frames_of_interest[a][3]), (255, 0, 0), 4) # debug line to check the default frames of interest for each finger
        
    
    #for c in range(len(HSV_ranges)):
        # comparison = cv2.addWeighted(frame, 0.7, cv2.cvtColor(stacked_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    sized_comparison = cv2.resize(frame, (width//2, height//2)) # resize the comparison image to fit in the screen, can be removed later to keep the original size for better visualization
    cv2.imshow(f"Frame with ROIs for each finger", sized_comparison)
    if cv2.waitKey(10) & 0xFF is ord('d'):
        # for each finger
        for c in fingers_to_track:
            a = c-1
            # for each marker of finger 'a'
            for b in range(len(corners[a])):
                focused_frame = copied_frame[corners[a][b][1]:corners[a][b][1]+ROI_height[a][b], corners[a][b][0]:corners[a][b][0]+ROI_width[a][b]]
                check_area_HSV_values(focused_frame, HSV_ranges,a, b, centroids[a][b][0], centroids[a][b][1], size=20)
                ShowRangeTester(focused_frame, HSV_ranges[b], preview=True)
                plt.close('all')

    print(f"Frame {count} processed") # debug line to check the processing of each frame
    
            
def insert_mask(frame, mask, corner_x, corner_y, default_frame, found_centroid, first_iteration):
    """
    Description:
        debug function
        insert the mask at the specified position in the given frame, to visualize the masked area in the full frame
    Parameters:
        frame (ndarray): The frame in which the mask should be inserted.
        mask (ndarray): The binary mask to be inserted into the frame.
        corner_x (int): The x-coordinate of the top left corner of the mask in the frame.
        corner_y (int): The y-coordinate of the top left corner of the mask in the frame.
        default_frame (tuple): The coordinates of the default frame of interest for the finger being processed, in the format (x1, y1, x2, y2).
        found_centroid (bool): A boolean indicating whether the centroid was found for the color being processed in the current frame. This is used to determine the color of the inserted mask (green if the centroid was found, blue otherwise).
        first_iteration (bool): A boolean indicating whether this is the first iteration of the frame processing loop.
    """
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # convert the mask to BGR format to be able to insert it in the frame
    mask_height, mask_width, channels = mask_bgr.shape
    frame_height, frame_width, frame_channels = frame.shape
    default_ROI_frame = frame[default_frame[1]:default_frame[3], default_frame[0]:default_frame[2]]

    if found_centroid and not first_iteration: 
        extended_to_default_mask = np.zeros_like(default_ROI_frame)
        top_limit = max(default_frame[1], corner_y)
        bottom_limit = min(default_frame[3], corner_y + mask_height)
        left_limit = max(default_frame[0], corner_x)
        right_limit = min(default_frame[2], corner_x + mask_width)
        

        cut_mask = mask_bgr[top_limit - corner_y:bottom_limit - corner_y, left_limit - corner_x:right_limit - corner_x]
        extended_to_default_mask[top_limit - default_frame[1]:bottom_limit - default_frame[1], left_limit - default_frame[0]:right_limit - default_frame[0]] = cut_mask

        modified_default_ROI = cv2.addWeighted(default_ROI_frame, 0.8, extended_to_default_mask, 0.5, 0) 
    else:
        modified_default_ROI = cv2.addWeighted(default_ROI_frame, 0.8, mask_bgr, 0.5, 0)
    
    # show the modified default ROI with the mask inserted
    #cv2.imshow(f"Default ROI with mask inserted", modified_default_ROI)
    #cv2.waitKey(0)
    frame[default_frame[1]:default_frame[3], default_frame[0]:default_frame[2]] = modified_default_ROI
    return frame

def check_area_HSV_values(frame, HSV_ranges, finger_index, color_index, center_x, center_y, size=5):
    """
    Description:
        debug function
        used to check the HSV values in the area around the specified position, 
        to check if the defined HSV ranges are correct for the colors we want to track, 
        and to adjust them if necessary
    Parameters:
        frame (ndarray): The frame in BGR format.
        finger_index (int): The index of the finger being processed
        color_index (int): The index of the color being processed
        center_x (int): The x-coordinate of the center of the area to check.
        center_y (int): The y-coordinate of the center of the area to check.
        size (int): The size of the area to check around the center position, in pixels.
    Returns:
        None
    Notes:

    """
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, HSV_ranges[color_index][0], HSV_ranges[color_index][1])
    
    comparison = cv2.addWeighted(frame, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0) # overlay the mask on the frame with some transparency
    cv2.rectangle(comparison, (center_x-size, center_y-size), (center_x+size, center_y+size), (255, 0, 0), 1) # show the area being checked
    cv2.imshow("detected pixels",cv2.resize(comparison, (600, 400))) # show the frame with the mask overlayed, resized for better visualization
    
    x = np.arange(center_x-size, center_x+size)
    y = np.arange(center_y-size, center_y+size)
    X, Y = np.meshgrid(x, y)
    hue = frame_hsv[center_y-size:center_y+size, center_x-size:center_x+size][:,:,0]
    InHueTtest = np.logical_and(hue >= HSV_ranges[color_index][0][0], hue <= HSV_ranges[color_index][1][0])
    print(f"    hue values in the area around the centroid for finger {finger_index}, marker {color_index}: {hue}")
    print(f"    in hue range: {InHueTtest}")

    inHueX = X.copy()    
    inHueY = Y.copy()
    inHueValues = hue.copy().astype(float)
    outHueValues = hue.copy().astype(float)

    inHueValues[~InHueTtest] = np.nan
    outHueValues[InHueTtest] = np.nan 
        
    saturation = frame_hsv[center_y-size:center_y+size, center_x-size:center_x+size][:,:,1]
    value = frame_hsv[center_y-size:center_y+size, center_x-size:center_x+size][:,:,2]

    # divide X,Y,hue,saturation and value by wether the (X,Y,Hue), (or respectively (X,Y,Saturation) or (X,Y,Value)) 
    # of the pixels in the area are in the defined HSV range for this color or not, to only show the values of the pixels that are detected in the mask

    #faut ajouter qqc pour la première itération, genre utiliser found_centroids
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4),subplot_kw={'projection': '3d'})
    
    
    #ax1.plot_wireframe(X, Y, hue, color='C2', label='points in range')
    ax1.plot_wireframe(X, Y, inHueValues, color='C0', label='in hue range')
    ax1.plot_wireframe(X, Y, outHueValues, color='C3', label='out of hue range')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.invert_yaxis()
    ax1.set_zlabel('Hue')
    ax1.set(xlim = (center_x-size, center_x+size), ylim = (center_y-size, center_y+size), zlim = (HSV_ranges[color_index][0][0], HSV_ranges[color_index][1][0]))
    ax1.legend()
    
    ax2.plot_wireframe(X, Y, saturation, color='g', label='Saturation')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Saturation')
    ax2.invert_yaxis()
    ax2.legend()

    ax3.plot_wireframe(X, Y, value, color='g', label='Value')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Value') 
    ax3.invert_yaxis()
    ax3.legend()
    
    fig.suptitle(f'HSV values in the area around the centroid for finger {finger_index}, marker {color_index}')
    plt.show(block=False)
    
def ShowRangeTester(image, HSV_ranges, preview=True):

    frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    range_filter = 'HSV'
    setup_trackbars(HSV_ranges)

    while True:
        
        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)

        thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        if preview:
            ok = cv2.bitwise_and(image, image, mask=thresh)
            show = cv2.addWeighted(ok,0.7,image,0.3,30)
            cv2.imshow("Preview", cv2.resize(show, (600, 400))) # resize the preview for better visualization, can be removed later to keep the original size for better visualization
        else:
            cv2.imshow("Original", image)
            cv2.imshow("Thresh", thresh)

        if cv2.waitKey(1) & 0xFF is ord(' '):
            cv2.destroyAllWindows()
            break

def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)
    cv2.resizeWindow("Trackbars", 400, 300) 


    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 1

        for j,val in enumerate(["H","S","V"]):
            cv2.createTrackbar("%s_%s" % (val, i), "Trackbars", range_filter[v][j], 255, callback)

def callback(value):
    pass

def get_trackbar_values(range_filter):
    values = []

    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)

    return values

def check_HSV_pixel_value(frame_hsv, x, y):
    """
    Description:
        debug function
        used to check the HSV value of a specific pixel in the frame
    Parameters:
        frame_hsv (ndarray): The frame in HSV format.
        x (int): The x-coordinate of the pixel to check.
        y (int): The y-coordinate of the pixel to check.
    Returns:
        hsv_value (tuple): The HSV value of the pixel at position (x, y).
    Notes:
        - This function is intended for debugging purposes and visualizing the HSV values of specific pixels in a frame.
    """
    return frame_hsv[x, y]

def stack_masks(masks):
    """
    Description:
        debug function
        returns the combined result of the masks given as input
    Parameters:
        masks (list of ndarray): A list of binary images (masks) for each color.
    Returns:
        combined_mask (ndarray): The combined result of the masks.
    Notes:
        - This function is intended for debugging purposes and visualizing the masks created for each color in the video frame.
    """
    combined_mask = np.zeros_like(masks[0])
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    return combined_mask
    
def add_markers_to_full_frame(frame, centroids, offset_x=0, offset_y=0):
    """ 
    Description:
        debug function
        returns the frame with circles drawn at the positions of the centroids of the masked areas (the colored markers).
    Parameters:
        frame (ndarray): The frame.
        centroids (list of tuples): A list of (x, y) coordinates for the centroids of the masked areas.
        offset_x (int): The x-coordinate offset to apply to the centroids.
        offset_y (int): The y-coordinate offset to apply to the centroids.
    Returns:
        frame (ndarray): The frame with markers drawn at the positions of the centroids.
    Notes:
        - This function is intended for debugging purposes and visualizing the positions of the colored markers in the video frame.
        - The offset parameters are used to adjust the position of the centroids because the masks are created on a region of the full image to make the processing faster
          if the function is used with the full frame, there is no need to input any offset
    """
    for centroid in centroids:
        if centroid is not None:
            cv2.circle(frame, (centroid[0] + offset_x, centroid[1] + offset_y), 15, (0, 255, 0), 2)
    return frame
       

def getAngle2(x1, y1, x2, y2):
    """
    Description:
        Calculate the angle between the horizontal axis and the line defined by the two points (x1, y1) and (x2, y2).
    Parameters:
        x1 (int): The x-coordinate of the first point.
        y1 (int): The y-coordinate of the first point.
        x2 (int): The x-coordinate of the second point.
        y2 (int): The y-coordinate of the second point.
    Returns:
        angle (float): The angle in degrees between the horizontal axis and the given line 
    """
    delta_x = x2 - x1
    delta_y = y2 - y1
    return math.degrees(math.atan2(delta_y, delta_x))

def orientAngle(angle, finger_id, phalanx_id):
    """
    Description:
        Orient the angle based on the finger and phalanx being processed, to ensure that the angle evolution is consistent with the expected movement of the fingers.
    Parameters:
        angle (float): The raw angle calculated between the horizontal axis and the line defined by the two markers
        finger_id (int): The ID of the finger being processed (from 1 to 6)
        phalanx_id (int): The ID of the phalanx being processed (1 for the first joint, 2 for the second joint)
    Returns:
        angle (float): The oriented angle, adjusted based on the finger and phalanx being processed"""
    
    if finger_id in [2, 3]:
        if phalanx_id == 2:
            # we offset by -90° to avoid crossing the 180° or -180° limit, which bugs the getAngleEvolution function when searching for the minimum angle
            return angle -90 
        else:
            return angle % 360
    elif finger_id in [5, 6]:
        return angle
    elif finger_id == 1: 
        return -angle #% 360
    else:
        return -angle

def getAngleDelta(angle1, angle2):
    """
    Description:
        Calculate the difference between two angles, taking into account the circular nature of angles (i.e., the fact that 0° and 360° are equivalent).
    Parameters:
        angle1 (float): The first angle in degrees.
        angle2 (float): The second angle in degrees.
    Returns:
        delta_angle (float): The difference between the two angles, in degrees, adjusted to be within the range [-180°, 180°].
    """
    return ((angle2 - angle1 + 180) % 360) - 180

def orientAnglev5(angle, finger_id, phalanx_id):
    """
    orientangle for markers selected on fingers 
    angle: computed by getAngle2()
    finger id: [1:6] id of the finger 
    """

    if phalanx_id == 1:
        if finger_id == 1:
            return - angle - 90
        if finger_id in [2,3]:
            return + angle - 90
        if finger_id == 4:
            return - angle + 90
        if finger_id in [5,6]:
            return + angle + 90
        else:
            print(f"invalid finger id ({finger_id}) received for orienting angle {angle}. returned default angle")
            return angle
        
    elif phalanx_id == 2:
        if finger_id == 1:
            return - (angle%360) + 180
        if finger_id in [2,3]:
            return + angle%360 - 180
        if finger_id == 4:
            return - angle
        if finger_id in [5,6]:
            return + angle
        else:
            print(f"invalid finger id ({finger_id}) received for orienting angle {angle}. returned default angle")
            return angle
    else:
        print(f"invalid phalanx id ({phalanx_id}) received for orienting angle {angle}. returned default angle")
        return angle

def getAnglesHistories(fingers_ids, points_histories, video = -1):
    """
    Description:
        Calculate the evolution of angles for the joints of the fingers being tracked, based on the history of centroids positions.
    Parameters:
        fingers_ids (list): A list of the IDs of the fingers to track (from 1 to 6)
        points_histories (list): A list containing the history of centroids positions for each color, for each finger, in the format [[[(x1, y1), (x2, y2), ...], [(x1, y1), (x2, y2), ...], ...], [...], ...]
                               This is the same format as the centroids_histories variable in the main code
    Returns:
        angles_histories (list): A list of lists, where each sublist contains the evolution of angles for the joints of a finger, in the format [[q1_angle_evolution, q2_angle_evolution, frame_index], [q1_angle_evolution, q2_angle_evolution, frame_index], ...], 
    """
    angles_histories = []
    for index, finger_id in enumerate(fingers_ids):
        angle_history = getAngleHistory(finger_id, points_histories[index], video=video) #q1_raw_angle_evolution, q2_raw_angle_evolution, frame_index
        angles_histories.append(angle_history)
    return angles_histories

def getAngleHistory(finger_id, points_history, offsets=(0,0), video = -1):
    """
    Description:
        Calculate the evolution of angles for two joints based on the history of centroids positions and 
    Parameters:
        finger_id (int): The ID of the finger for which to calculate the angle evolution (from 1 to 6)
        points_history (list): A list containing the history of centroids positions for each color, in the format [[(x1, y1), (x2, y2), ...], [(x1, y1), (x2, y2), ...], ...]
                                  This is not the same format as the centroids_history variable in the main code
                                  the one used here lacks the first level for the fingers
        offsets (tuple): A tuple containing the angle offsets for the two joints (q1_offset, q2_offset), 
                        the offsets are determined on the first frame as the difference between the marker based angles and the user selected reference angles
        video: the index of the video used when calling this function
                set to -1 by default if the function is used on a single video

    Returns:
        q1_angle_history (list): A list of angles for the first joint (between the horizontal segment passing by the first marker and the segment between the second and third marker)
        q2_angle_history (list): A list of angles for the second joint (between the segment passing by the second and third marker and the segment passing by the fourth and fifth marker)
        frame_index (list): A list of the frame indexes corresponding to the calculated angles, to be able to plot the angle evolution over time

    """
    
    q1_angle_history = np.array([])
    q2_angle_history = np.array([])
    frame_index = []
    
    for i in range(len(points_history[0])): # iterate through the positions of the markers
        phal1p1 = (points_history[-4][i][0], points_history[-4][i][1])
        phal1p2 = (points_history[-3][i][0], points_history[-3][i][1])
        phal2p1 = (points_history[-2][i][0], points_history[-2][i][1])
        phal2p2 = (points_history[-1][i][0], points_history[-1][i][1])
        if None in [phal1p1[0], phal1p1[1], phal1p2[0], phal1p2[1], phal2p1[0], phal2p1[1], phal2p2[0], phal2p2[1]]: # if one of the centroids is not detected in this frame, we skip the angle calculation for this frame
            print(f"  Warning: missing centroid for finger {finger_id} at frame {i}, skipping angle calculation for this frame")
            continue #TODO modifier pour plus tard, pour ne pas décaler les angles calculés pour les autres doigts
            # so far ça devrait pas créer de problème vu que si on rate un marqueur on réajuste les ranges hsv
        
        phal1_angle = orientAnglev5(getAngle2(phal1p1[0], phal1p1[1], phal1p2[0], phal1p2[1]), finger_id, 1)
        phal2_angle = orientAnglev5(getAngle2(phal2p1[0], phal2p1[1], phal2p2[0], phal2p2[1]), finger_id, 2)

        q1_angle = (((phal1_angle - offsets[0]) + 90) % 360) - 90
        q2_angle = (((phal2_angle - phal1_angle - offsets[1]) + 90) % 360) - 90

        angle1_change = q1_angle - q1_angle_history[-1] if len(q1_angle_history) > 0 else 0
        angle2_change = q2_angle - q2_angle_history[-1] if len(q2_angle_history) > 0 else 0
        
        if abs(angle1_change) > 100 and abs(angle1_change) < 270:
            print(f"  Warning: [video{video}] sudden change for finger {finger_id} in angle q1: from {q1_angle_history[-1] if len(q1_angle_history) > 0 else phal1_angle:.2f} to {phal1_angle:.2f}, change = {angle1_change:.2f}")
            phal1_angle = q1_angle_history[-1] if len(q1_angle_history) > 0 else q1_angle 
        if abs(angle2_change) > 100 and abs(angle2_change) < 270:
            print(f"  Warning: [video{video}] sudden change for finger {finger_id} in angle q2: from {q2_angle_history[-1] if len(q2_angle_history) > 0 else phal2_angle - phal1_angle:.2f} to {phal2_angle - phal1_angle:.2f}, change = {angle2_change:.2f}")
            phal2_angle = phal1_angle + (q2_angle_history[-1] if len(q2_angle_history) > 0 else q2_angle )

        q1_angle_history = np.append(q1_angle_history, q1_angle)
        q2_angle_history = np.append(q2_angle_history, q2_angle)
        frame_index.append(i)

    return q1_angle_history, q2_angle_history, frame_index

def extract_peak_angles(angle_evolution, frame_index):
    """
    Description:
        Extract the angles at the peaks of each period, to analyse a possible change in reach
    Parameters:
        angle_evolution (list): A list of angles for a joint over time.
        frame_index (list): A list of the frame indexes corresponding to the angles in angle_evolution.
        distance (int): The minimum distance between peaks, in number of frames, to avoid detecting multiple peaks for the same period.
        frames_per_segment (int): The number of frames in each segment of the movement, used to calculate the positions of the seams between segments.
        seam_margin (int): The minimum distance in number of frames between a detected peak and a seam, to avoid detecting peaks that are too close to the seams, where the movement is not smooth and the angle evolution is not reliable.
    Returns:
        peak_angles (list): A list of angles at the detected peaks.
        peak_frames (list): A list of the frame indexes corresponding to the detected peaks.
    """
    # use this for 30 fps, 331 motor rpm
    foundpeaks, _ = find_peaks(angle_evolution, distance=8, prominence=15)
    # for slower movements, we need to increase the x distance between peaks to avoid detecting multiple pooints from the same period
    #foundpeaks, _ = find_peaks(angle_evolution, distance=50, prominence=10) # distance = 30 frames = 1 second, prominence = 10 degrees
    peaks = foundpeaks[1:-1]

    peak_angles = angle_evolution[peaks]
    peak_frames = [frame_index[i] for i in peaks]
    return peak_angles, peak_frames

def extract_valley_angles(angle_evolution, frame_index):
    """
    Description:
        Extract the angles at the valleys of each period, to analyse a possible change in reach
        for more details, see the description of the extract_peak_angles function.
        this does the same thing but for the valleys instead of the peaks so we use the negative of the angle evolution
    """
    peak_angles, peak_frames = extract_peak_angles(-angle_evolution, frame_index)
    return -peak_angles, peak_frames

def get_peak_angles_histories(fingers_ids, AnglesHistories):
    peak_angles_evolutions = []
    for index in range(len(fingers_ids)):
        q1_raw_angle_evolution, q2_raw_angle_evolution, frame_index = AnglesHistories[index]
        q1_peak_angles, q1_peak_frames = extract_peak_angles(q1_raw_angle_evolution, frame_index)
        q2_peak_angles, q2_peak_frames = extract_peak_angles(q2_raw_angle_evolution, frame_index)
        peak_angles_evolutions.append((q1_peak_angles, q1_peak_frames, q2_peak_angles, q2_peak_frames))
    return peak_angles_evolutions

def get_valley_angles_histories(fingers_ids, AnglesHistories):
    """
    Description:
        Get the angles at the valleys of each period for each finger, based on the history of angles for each finger.
    Parameters:
        fingers_ids (list): A list of the IDs of the fingers to track (from 1 to 6)
        AnglesHistories (list): A list of lists, where each sublist contains the evolution of angles for the joints of a finger,
                in the format [[q1_angle_evolution, q2_angle_evolution, frame_index], [q1_angle_evolution, q2_angle_evolution, frame_index], ...],
                as returned by the getAnglesHistories function.
    """
    valley_angles_evolutions = []
    for index in range(len(fingers_ids)):
        q1_raw_angle_evolution, q2_raw_angle_evolution, frame_index = AnglesHistories[index]
        q1_valley_angles, q1_valley_frames = extract_valley_angles(q1_raw_angle_evolution, frame_index)
        q2_valley_angles, q2_valley_frames = extract_valley_angles(q2_raw_angle_evolution, frame_index)
        valley_angles_evolutions.append((q1_valley_angles, q1_valley_frames, q2_valley_angles, q2_valley_frames))
    return valley_angles_evolutions

def get_range_angles_histories(fingers_ids, peak_angles_histories, valley_angles_histories):
    range_histories = []
    for i in range(len(fingers_ids)): # pour chaque doigt i traqué
        finger_range_evol = []
        for j in range(2): # pour chaque articulation j du doigt i
            n_periods = min(len(peak_angles_histories[i][2*j]), len(valley_angles_histories[i][2*j]))
            q_range = []
            q_pos = []
            for k in range(n_periods): # pour chaque période détecté
                q_range.append(float(peak_angles_histories[i][2*j][k])-float(valley_angles_histories[i][2*j][k]))
                q_pos.append(peak_angles_histories[i][2*j+1][k])
            finger_range_evol.extend((q_range,q_pos))
        range_histories.append(finger_range_evol)
    #(q1_range_vals, q1_range_frames), (q2_range_vals, q2_range_frames) = range_histories[0]
    return range_histories

def plot_angle_evolution(fingers_ids, angles_histories):
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'pink']
    nrows, ncols = get_subplot_layout(len(fingers_ids))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows), sharex=False, sharey=False)
    
    # normaliser axes en tableau 1D pour itérer facilement
    axes_flat = np.array(axes).flatten()

    for index, finger in enumerate(fingers_ids):
        ax = axes_flat[index]
        q1_raw_angle_evolution, q2_raw_angle_evolution, frame_index = angles_histories[index]

        ax.plot(frame_index, q1_raw_angle_evolution, color=colors[0], marker='.', markersize=1, linestyle='', alpha=0.7, label=f"finger {finger} q1")
        ax.plot(frame_index, q2_raw_angle_evolution, color=colors[2], marker='.', markersize=1, linestyle='', alpha=0.7, label=f"finger {finger} q2")
        ax.set_title(f"Finger {finger}")
        ax.set_xlabel("seconds")
        ax.set_ylabel("Angle (degrees)")
        #ax.set_ylim(0, 110)
        ax.grid()
        ax.legend(loc="upper right")

    # cacher les subplots vides
    for idx in range(len(fingers_ids), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Evolution of Joint Angles Over Time for Fingers {fingers_ids}")
    plt.tight_layout()

def plot_peak_angles_evolution(fingers_ids, peak_angles_histories):
    """
    Description:
        display function
        Plot the evolution of angles at the peaks of each period for the given fingers based on the history of centroids positions.
    Parameters:
        fingers_ids (list): A list of finger IDs for which to plot the peak angle evolution (from 1 to 6).
        peak_angles_histories (list): A list as returned by the get_peak_angles_histories function,
                containing the history of angles at the peaks of each period for each finger,
                in the format [[(q1_peak_angles, q1_peak_frames, q2_peak_angles, q2_peak_frames)], [...], ...]
    returns:
        None
    """
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'pink']
    nrows, ncols = get_subplot_layout(len(fingers_ids))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows), sharex=False, sharey=False)
    
    # normaliser axes en tableau 1D pour itérer facilement
    axes_flat = np.array(axes).flatten()

    for index,fingerlist in enumerate(peak_angles_histories):
        ax = axes_flat[index]
        q1_peak_angles, q1_peak_frames, q2_peak_angles, q2_peak_frames = fingerlist
        ax.plot(q1_peak_frames, q1_peak_angles, color = colors[0],marker='.', markersize=1,linestyle='', alpha = 0.7, label=f"finger {fingers_ids[index]} q1 angle at peaks")
        ax.plot(q2_peak_frames, q2_peak_angles, color = colors[2],marker='.', markersize=1,linestyle='', alpha = 0.7, label=f"finger {fingers_ids[index]} q2 angle at peaks")
        ax.set_xlabel("Seconds")
        ax.set_ylabel("Angle (degrees)")
        ax.set_title(f" for Fingers {fingers_ids[index]} at peaks")
        ax.grid()
        #ax.set_ylim(0, 110)
        ax.legend(loc="upper right")

    #cacher les subplots vides
    for idx in range(len(fingers_ids), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Evolution of Joint Peak Angles Over Time")
    plt.tight_layout()
    

def plot_valley_angles_evolution(fingers_ids, valley_angles_histories):
    """
    Description:
        display function
        Plot the evolution of angles at the valleys of each period for the given fingers based on the history of angles.
    Parameters:
        fingers_ids (list): A list of finger IDs for which to plot the valley angle evolution (from 1 to 6).
        valley_angles_histories (list): A list as returned by the get_valley_angles_histories functionµ
                containing the history of angles at the valleys of each period for each finger,
                in the format [[(q1_valley_angles, q1_valley_frames, q2_valley_angles, q2_valley_frames)], [...], ...]
    returns:
        None
    """
    colors = ['blue','orange','green','red','purple','pink']

    nrows, ncols = get_subplot_layout(len(fingers_ids))
    fig, subs = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows), sharex=False, sharey=False)

    # normaliser axes en tableau 1D pour itérer facilement
    subs_flat = np.array(subs).flatten()    

    for index,fingerlist in enumerate(valley_angles_histories):
        q1_valley_angles, q1_valley_frames, q2_valley_angles, q2_valley_frames = fingerlist
        sub = subs_flat[index]
        sub.plot(q1_valley_frames, q1_valley_angles, color = colors[0],marker='.', markersize=1,linestyle='', alpha = 0.7, label=f"finger {fingers_ids[index]} q1 valley angles")
        sub.plot(q2_valley_frames, q2_valley_angles, color = colors[2],marker='.', markersize=1,linestyle='', alpha = 0.7, label=f"finger {fingers_ids[index]} q2 valley angles")
        sub.set_xlabel("Seconds")
        sub.set_ylabel("Angle (degrees)")
        sub.set_title(f"Finger {fingers_ids[index]}")
        sub.grid()
        #sub.set_ylim(0, 110)
        sub.legend(loc="upper right")

    #cacher les subplots vides
    for idx in range(len(fingers_ids), len(subs_flat)):
        subs_flat[idx].set_visible(False)
    
    fig.suptitle(f"Evolution of Joint Valley Angles Over Time")
    plt.tight_layout()

def plot_range_histories(fingers_ids, range_angles_histories):
    """
    Description:
        display function
        Plot the evolution of angles ranges for the joints of the fingers being tracked, based on the history of angles at the peaks and valleys of each period.
    Parameters:
        fingers_ids (list): A list of finger IDs for which to plot the angle ranges evolution (from 1 to 6).
        range_angles_histories (list): A list as returned by the get_range_angles_histories function,
                containing the history of angles at the peaks and valleys of each period for each finger,
                in the format [[((q1_peak_angles, q1_peak_frames), (q2_peak_angles, q2_peak_frames)), ((q1_valley_angles, q1_valley_frames), (q2_valley_angles, q2_valley_frames))], [...], ...]
    returns:
        None
    """
    colors = ['blue','orange','green','red','purple','pink']
    nrows, ncols = get_subplot_layout(len(fingers_ids))
    fig, subs = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows), sharex=False, sharey=False)
    subs_flat = np.array(subs).flatten()

    for index,fingerlist in enumerate(range_angles_histories):
        q1_range_vals, q1_range_frames, q2_range_vals, q2_range_frames = fingerlist
        sub = subs_flat[index]
        sub.plot(q1_range_frames, q1_range_vals, color = colors[0],marker='.', markersize=1,linestyle='', alpha = 0.7, label=f"finger {fingers_ids[index]} q1 angles ranges ")
        sub.plot(q2_range_frames, q2_range_vals, color = colors[2],marker='.', markersize=1,linestyle='', alpha = 0.7, label=f"finger {fingers_ids[index]} q2 angles ranges ")
        sub.set_xlabel("Seconds")
        sub.set_ylabel("Angle (degrees)")
        sub.set_title(f"Finger {fingers_ids[index]}")
        sub.grid()
        #sub.set_ylim(0, 110)
        sub.legend(loc="upper right")
    
    #cacher les subplots vides
    for idx in range(len(fingers_ids), len(subs_flat)):
        subs_flat[idx].set_visible(False)
    fig.suptitle(f"Evolution of Joint Angles Ranges Over Time")
    plt.tight_layout()


def plot_markers_histories(points_history, fingers_to_plot, Markers_colors, frame_width, frame_height):
    """
    Description:
        display function
        plot the history of centroids positions in a X-Y frame, for each color, for each finger, to visualize the recorded positions of the markers over time
    Parameters:
        fingers_to_plot (list): A list of finger IDs for which to plot the markers histories (from 1 to 6).
        points_history (list): A list as returned by the centroids_histories variable in the main code,
            containing the history of centroids positions for each color, for each finger,
            in the format [[[(x1, y1), (x2, y2), ...], [(x1, y1), (x2, y2), ...], ...], [...], ...]
    returns:
        None
    """
    plt.figure(figsize=(10, 5))
    for e,i in enumerate(fingers_to_plot):
        for j in range(len(Markers_colors)):
            x = [pos[0] for pos in points_history[e][j]]
            y = [pos[1] for pos in points_history[e][j]]
            plt.scatter(x, y, s=2, label=f"finger {i} marker {Markers_colors[j]}")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title(f"History of Centroids Positions for Fingers {fingers_to_plot}") 
    plt.grid()
    plt.xlim(0, frame_width)
    plt.ylim(0, frame_height)
    plt.gca().invert_yaxis() # invert the y-axis to have the same orientation as the video frame
    #plt.legend(loc='upper center')

def get_subplot_layout(n_fingers):
    """
    Description:
        display helper function
        Calculate the number of rows and columns for the subplots based on the number of fingers being tracked
    Parameters:
        n_fingers (int): The number of fingers being tracked.
    Returns:
        nrows (int): The number of rows for the subplots.
        ncols (int): The number of columns for the subplots.
    Notes:
        6 or 5 fingers: 3 rows, 2 columns
        4 or 3 fingers: 2 rows, 2 columns
        2 fingers: 1 row, 2 columns
        1 finger: 1 row, 1 column
    """
    ncols = 1 if n_fingers <= 2 else 2
    nrows = (n_fingers + ncols - 1) // ncols  # ceiling division
    return nrows, ncols



