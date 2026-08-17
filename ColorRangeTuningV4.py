"""
============================================================================================================
README
ceci est la dernière version du code
Ce code sert à tuner les plages HSV des différentes couleurs pour la détection des marqueurs de couleur.

UTILISATION:
    -
TODO:
    - tout mettre en une seule fenetre?
    - regarder pour mettre la trackbar window en plusieurs colonnes
- 
Sources ayant servi de base pour ce code:
- https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
- https://stackoverflow.com/questions/44588279/find-and-draw-the-largest-contour-in-opencv-on-a-specific-color-python
- https://stackoverflow.com/questions/28327020/opencv-detect-mouse-position-clicking-over-a-picture
============================================================================================================
"""

#
# ===== Imports =====############################################################################################################
#

from AllVideoFunctions import *
import threading

#
# ===== variables =====############################################################################################################
#

# ── Settings ────────────────────────────────────────────────────────
testName = "RunBlancs" # Update the color_ranges variable in the AllVideoFunctions.py after running this code
# pas de marqueurs sur les videos RunBase
video_path = "Runs\\FatigueTests\\RunBaseLowAngle\\test_106\\video_shift01_2026-03-06_17-39-04.h264"
video_path = "Runs\\FatigueTests\\RunQuartSpeed\\test_151\\video_shift01_2026-05-11_19-50-08.h264"
#video_path = "Runs\\FatigueTests\\RunQuartSpeed\\test_151\\video_shift80_2026-05-12_09-00-17.h264"
# pas de marqueurs sur les videos RunBaseUltimaker
video_path = "Runs\\FatigueTests\\Runbleus\\test_124\\video_shift01_2026-03-25_12-42-43.h264"
video_path = "Runs\\FatigueTests\\Runtpu\\test_121\\video_shift01_2026-03-24_16-09-12.h264"
#video_path = "Runs\\FatigueTests\\RunClimBAseModel\\test_140\\video_shift01_2026-04-27_20-19-21.h264"
#video_path = "Runs\\FatigueTests\\RunClimBAseModel\\test_140\\video_shift39_2026-04-28_02-39-33.h264"
#video_path = "Runs\\FatigueTests\\RunClimModded\\test_144\\video_shift01_2026-04-29_15-46-19.h264"
#video_path = "Runs\\FatigueTests\\RunClimTPU\\test_147\\video_shift477_2026-05-07_22-02-40.h264"

video_path = video_paths[testName]



# ── Other variables ────────────────────────────────────────────────────────
scalingFactor = 2.7 # ratio between the normal video resolution and the displayed complete image resolution
zoomSize = 20 # size of the zoomed image in the displayed complete image (-> size of the zoomed image in the normal video is zoomSize*scalingFactor)
#
# ======== functions ======############################################################################################################
#

def Feur(x):
    """
    dummy function for trackbar callback, does nothing
    """
    pass


def create_trackbarsWindow(HSV_range, kernel_size=3, min_area=100):
    """
    this function creates a window with trackbars to adjust the parameters of the color detection algorithm
    (HSV min and max values, kernel size for morphological operations, and minimum area for contour detection)
    params:
        - HSV_range: a tuple of two numpy arrays, the first one is the lower bound of the HSV range and the second one is the upper bound
            ex: (HSV_range = np.array([0, 65, 169]), np.array([14, 255, 255]))
        - kernel_size: the initial value for the kernel size trackbar
        - min_area: the initial value for the minimum area trackbar
    return:
        - nothing, but creates a window
    """
    cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("controls", 320, 360)

    #trackbars pour ajuster les valeurs HSV
    cv2.createTrackbar("H min", "controls", HSV_range[0][0], 179, Feur)
    cv2.createTrackbar("H max", "controls", HSV_range[1][0], 179, Feur)
    cv2.createTrackbar("S min", "controls", HSV_range[0][1], 255, Feur)
    cv2.createTrackbar("S max", "controls", HSV_range[1][1], 255, Feur)
    cv2.createTrackbar("V min", "controls", HSV_range[0][2], 255, Feur)
    cv2.createTrackbar("V max", "controls", HSV_range[1][2], 255, Feur)

    # Trackbars pour nettoyage du masque
    cv2.createTrackbar("kernel", "controls", kernel_size, 20, Feur)
    cv2.createTrackbar("min area", "controls", min_area, 5000, Feur)

def setTrackbarsValues(HSV_range):
    """
    this function sets the trackbar positions to the values of the given HSV range
    params:
    - HSV_range: a tuple of two numpy arrays, the first one is the lower bound of the HSV range and the second one is the upper bound
        ex: (HSV_range = np.array([0, 65, 169]), np.array([14, 255, 255]))
    """
    cv2.setTrackbarPos("H min", "controls", HSV_range[0][0])
    cv2.setTrackbarPos("H max", "controls", HSV_range[1][0])
    cv2.setTrackbarPos("S min", "controls", HSV_range[0][1])
    cv2.setTrackbarPos("S max", "controls", HSV_range[1][1])
    cv2.setTrackbarPos("V min", "controls", HSV_range[0][2])
    cv2.setTrackbarPos("V max", "controls", HSV_range[1][2])

def PushHSVLimits(range_name, current_range, new_point):
    """
    This function expands the current HSV range up to the new value if the new value is outside the current range.
    It takes into account the special case of the hue range when it wraps around 180°.
    params:
    - range_name: a string to identify the range being updated (e.g. "trackbar", "whitelisted", "blacklisted")
    - current_range: a tuple of two numpy arrays representing the current HSV range
    - new_point: a numpy array representing the new HSV value that we want to include in the range
    return:
        - a tuple of two numpy arrays representing the updated HSV range
    """
    h, s, v = new_point
    min_h, min_s, min_v = current_range[0]
    max_h, max_s, max_v = current_range[1]
    min_h, max_h = increaseCyclicRange(f"{range_name} H", (min_h, max_h), h,180)
    min_s, max_s = increaseRange(f"{range_name} S", (min_s, max_s), s)
    min_v, max_v = increaseRange(f"{range_name} V", (min_v, max_v), v)
    return (np.array([min_h, min_s, min_v]), np.array([max_h, max_s, max_v]))

def increaseRange(range_name,current_range, new_value):
    """
    this function expands the current range limits to the new value if the new value is outside the current range
    """
    current_min_val = int(current_range[0])
    current_max_val = int(current_range[1])
    new_value = int(new_value)
    if not isInRange(new_value, current_range):
        dist_to_min = abs(current_min_val - new_value)
        dist_to_max = abs(new_value - current_max_val)
        if dist_to_min < dist_to_max:
            print(f"---reduced {range_name} min from {current_min_val} to {new_value}")
            current_min_val = new_value
        else:
            print(f"---increased {range_name} max from {current_max_val} to {new_value}")
            current_max_val = new_value
    return (current_min_val, current_max_val)

def increaseCyclicRange(range_name,current_range, new_value, max_value=180):
    """
    this function expands the current range limits to the new value if the new value is outside the current range
    si la plage de teinte enveloppe max_value, la fonction doit en tenir compte 
    """
    current_min_val = int(current_range[0])
    current_max_val = int(current_range[1])
    new_value = int(new_value)
    if not isInCyclicRange(new_value, current_range):
        dist_to_min = CyclicDist(current_min_val, new_value, max_value)
        dist_to_max = CyclicDist(current_max_val, new_value, max_value)
        if dist_to_min < dist_to_max:
            print(f"---reduced {range_name} min from {current_min_val} to {new_value}")
            current_min_val = new_value
        else:
            print(f"---increased {range_name} max from {current_max_val} to {new_value}")
            current_max_val = new_value
    return (current_min_val, current_max_val)

def reduceRange(range_name,current_range, new_value):
    """
    This function reduces the current range limits to exclude the new value
    """
    current_min_val = int(current_range[0])
    current_max_val = int(current_range[1])
    new_value = int(new_value)
    if isInRange(new_value, current_range):
        dist_to_min = abs(new_value - current_min_val)
        dist_to_max = abs(current_max_val - new_value)
        if dist_to_min < dist_to_max:
            print(f"---increased {range_name} min from {current_min_val} to {new_value}")
            current_min_val = (new_value + 1)
        else:
            print(f"---reduced {range_name} max from {current_max_val} to {new_value}")
            current_max_val = (new_value - 1)
    return (current_min_val, current_max_val)

def reducecyclicRange(range_name,current_range, new_value, max_value=180):
    """
    This function reduces the current range limits to exclude the new value
    si la plage de teinte enveloppe max_value, la fonction doit en tenir compte 
    """
    current_min_val = int(current_range[0])
    current_max_val = int(current_range[1])
    new_value = int(new_value)
    if isInCyclicRange(new_value, current_range):
        dist_to_min = CyclicDist(current_min_val, new_value, max_value)
        dist_to_max = CyclicDist(current_max_val, new_value, max_value)
        if dist_to_min < dist_to_max:
            print(f"---increased {range_name} min from {current_min_val} to {new_value}")
            current_min_val = (new_value + 1) % max_value
        else:
            print(f"---reduced {range_name} max from {current_max_val} to {new_value}")
            current_max_val = (new_value - 1) % max_value
    return (current_min_val, current_max_val)

def isInRange(value,range):
    """
    vérifie si value appartient à l'ensemble spécifié. 
    cette fonction retournera False pour les ensemble où max < min
    params:
    -   value: (int) the value to be tested
    -   range: (int,int) the range to test value
    return:
        True if value is in the specified range, False otherwise
    """
    min_val, max_val = range
    return min_val <= value <= max_val

def isInCyclicRange(value,range):
    """
    vérifie si value appartient à l'ensemble spécifié. 
    si la plage de teinte enveloppe la valeur maximale, la fonction doit en tenir compte pour déterminer si value est dans la plage ou pas
    params:
    -   value: (int) the value to be tested
    -   range: (int,int) the range to test value
    return:
        True if value is in the specified range, False otherwise
    """
    min_val, max_val = range
    if min_val < max_val:
        return min_val <= value <= max_val
    else:
        return value >= min_val or value <= max_val

def getMedian(a,b,max_value=180):
    """
    retourne la valeur a mi-chemin de a vers b.
    Si a > b, la fonction considèrera un retour à zéro au dessus de max_value
    """
    a = int(a)
    b = int(b)
    if a <= b:
        return int(a + (b-a)/2)
    else:
        return int((a + (max_value-a + b)/2) % max_value)

def getAntiMedian(a,b,max_value):
    a = int(a)
    b = int(b)
    max_value = int(max_value)
    b_to_max = max_value-b
    if a > b_to_max:
        return 0
    return max_value - 1

def getCyclicrangeAntimedian(a,b,max_value=180):
    return (getMedian(a,b)+max_value//2)%max_value

def CyclicDist(a,b,max_value=180):
    """
    calcule la distance minimale séparant 2 points, considérant qu'on retourne à zéro apres max_value
    params:
    a,b: (int) les 2 points dont on veut calculer la distance
    return:
        (int) la distance minimale entre les 2 points
    """
    return min(abs(a-b), max_value-abs(a-b))


def fetch_HSV_Values(event, x, y, flags, param):
    global mouseX, mouseY
    global goodHSVValue, badHSVValue
    global zoomtocoords
    global pending_click_timer

    def do_single_click(x, y):
        global mouseX, mouseY, zoomtocoords
        mouseX, mouseY = x, y
        zoomtocoords = True
        print(f"zoomed on picture at ({mouseX}, {mouseY})")

    if event == cv2.EVENT_LBUTTONDBLCLK:
        if pending_click_timer is not None:
            pending_click_timer.cancel()  # annule le single click en attente
        mouseX, mouseY = x, y
        goodHSVValue = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)[mouseY, mouseX]
        print(f"HSV Good value at ({mouseX}, {mouseY}): {goodHSVValue}")

    elif event == cv2.EVENT_RBUTTONDBLCLK:
        mouseX, mouseY = x, y
        badHSVValue = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)[mouseY, mouseX]
        print(f"HSV Bad value at ({mouseX}, {mouseY}): {badHSVValue}")

    elif event == cv2.EVENT_LBUTTONDOWN:
        pending_click_timer = threading.Timer(0.4, do_single_click, args=(x, y))
        pending_click_timer.start()
    return x,y

def fetch_HSV_Values_zoomed(event, x, y, flags, param):
    global goodHSVValue, badHSVValue
    # param = (resized_frame, posx, posy, zoomSize)
    resized_frame, posx, posy, zoomSize = param

    if event == cv2.EVENT_LBUTTONDBLCLK:
        # x,y sont dans l'image 300x300 → reconvertir vers coordonnées resized_frame
        real_x = int(posx - zoomSize + x * (zoomSize * 2) / 300)
        real_y = int(posy - zoomSize + y * (zoomSize * 2) / 300)
        real_x = max(0, min(real_x, resized_frame.shape[1] - 1))
        real_y = max(0, min(real_y, resized_frame.shape[0] - 1))
        hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
        goodHSVValue = hsv_frame[real_y, real_x]
        print(f"HSV Good value (zoomed) at ({real_x}, {real_y}): {goodHSVValue}")

    elif event == cv2.EVENT_RBUTTONDBLCLK:
        real_x = int(posx - zoomSize + x * (zoomSize * 2) / 300)
        real_y = int(posy - zoomSize + y * (zoomSize * 2) / 300)
        real_x = max(0, min(real_x, resized_frame.shape[1] - 1))
        real_y = max(0, min(real_y, resized_frame.shape[0] - 1))
        hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
        badHSVValue = hsv_frame[real_y, real_x]
        print(f"HSV Bad value (zoomed) at ({real_x}, {real_y}): {badHSVValue}")


############################################################################################################
#
## ======== Main ========
#
############################################################################################################


# initialisation
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
width = frame.shape[1]
height = frame.shape[0]
Markers_colors = TestMarkers[testName]
HSV_ranges = [(color_ranges[Markers_colors[i]][0], color_ranges[Markers_colors[i]][1]) for i in range(len(Markers_colors))]
i = 0 # index du marker en cours de tuning
j = 0 # index de la frame en cours de lecture
mouseX, mouseY = 0, 0
goodHSVValue = (-1, -1, -1)
badHSVValue = (-1, -1, -1)
pending_click_timer = None
frameSize = [int(frame.shape[1]//scalingFactor), int(frame.shape[0]//scalingFactor)]
zoomtocoords = False
posy = frameSize[1]//2
posx = frameSize[0]//2

h = getMedian(HSV_ranges[i][0][0],HSV_ranges[i][1][0])
s = getMedian(HSV_ranges[i][0][1],HSV_ranges[i][1][1])
v = getMedian(HSV_ranges[i][0][2],HSV_ranges[i][1][2])

goodHSVRange = (np.array([h, s, v]), np.array([h+1, s+1, v+1]))
goodHSVRange_history = []  # stack d'états (goodHSVRange, min_h, max_h, min_s, max_s, min_v, max_v)
print(f"\nTesting color range: lower={HSV_ranges[i][0]}, upper={HSV_ranges[i][1]}")

#stocker les valeurs HSV apres ajustements dans une liste pour les afficher à la fin
new_plages_HSV = [(np.array([180, 255, 255]), np.array([0, 0, 0])) for _ in range(len(HSV_ranges))]

# création de l'UI de tuning des plages
create_trackbarsWindow(HSV_ranges[0], kernel_size, min_area)

#boucle principale pour lire la vidéo
while True:
    # lire sliders
    hmin = cv2.getTrackbarPos("H min", "controls")
    hmax = cv2.getTrackbarPos("H max", "controls")
    smin = cv2.getTrackbarPos("S min", "controls")
    smax = cv2.getTrackbarPos("S max", "controls")
    vmin = cv2.getTrackbarPos("V min", "controls")
    vmax = cv2.getTrackbarPos("V max", "controls")
    kernel_size = cv2.getTrackbarPos("kernel", "controls")
    min_area = cv2.getTrackbarPos("min area", "controls")
    lower = np.array([hmin, smin, vmin])
    upper = np.array([hmax, smax, vmax])

    # masque HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if hmin > hmax:  # si la plage de teinte enveloppe 180°
        lower1 = np.array([0, smin, vmin])
        upper1 = np.array([hmax, smax, vmax])
        lower2 = np.array([hmin, smin, vmin])
        upper2 = np.array([179, smax, vmax])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, lower, upper)

    # nettoyage morphologique
    if kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    resized_frame = cv2.resize(frame, (frameSize[0], frameSize[1]))
    resised_mask = cv2.resize(mask, (frameSize[0], frameSize[1]))

    # trouver les blobs
    contours, _ = cv2.findContours(resised_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    resized_output = resized_frame.copy()
    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # a bit of drawing
        cv2.drawContours(resized_output, [cnt], -1, (0,255,0), 3)

        # centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(resized_output, (cx,cy), 3, (0,0,255), -1)
           
   # display
    converted_mask = cv2.cvtColor(resised_mask, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(resized_output, 0.7, converted_mask, 0.5, 0)
    combined = np.hstack((overlay,converted_mask))
    cv2.imshow(f"Color Detection for marker {i}", combined)

    # get hsv values on double click
    cv2.setMouseCallback(f"Color Detection for marker {i}",fetch_HSV_Values, resized_frame)
    first_time = False

    if zoomtocoords == True:
        # avoid top and bottom
        posy = max(mouseY,(zoomSize//2)*scalingFactor)
        posy = min(posy,frameSize[1] - (zoomSize//2)*scalingFactor)
        # avoid left and right
        posx = max(mouseX,(zoomSize//2)*scalingFactor)
        posx = min(posx,frameSize[0] - (zoomSize//2)*scalingFactor)

        zoomtocoords = False
        print("\n")

    cutpictureu = frame[int((posy - zoomSize)*scalingFactor): int((posy+zoomSize)*scalingFactor), int((posx-zoomSize)*scalingFactor): int((posx+zoomSize)*scalingFactor)]
    cutmaskeu = mask[int((posy - zoomSize)*scalingFactor): int((posy+zoomSize)*scalingFactor), int((posx-zoomSize)*scalingFactor): int((posx+zoomSize)*scalingFactor)]
    cutmasked = cv2.bitwise_and(cutpictureu, cutpictureu, mask = cutmaskeu)
    blended_cut = cv2.addWeighted(cutmasked, 0.7, cutpictureu, 0.3, 30)
    zoomedpicture = cv2.resize(blended_cut, (300, 300))
    cv2.imshow("zoomed picture", zoomedpicture)

    # get HSV values on double click on zoomed picture
    cv2.setMouseCallback("zoomed picture", fetch_HSV_Values_zoomed, (resized_frame, posx, posy, zoomSize))

    #update tracbars with mouse HS  V values
    if goodHSVValue[0] != -1 or badHSVValue[0] != -1:
        
        min_h = cv2.getTrackbarPos("H min", "controls")
        max_h = cv2.getTrackbarPos("H max", "controls")
        min_s = cv2.getTrackbarPos("S min", "controls")
        max_s = cv2.getTrackbarPos("S max", "controls")
        min_v = cv2.getTrackbarPos("V min", "controls")
        max_v = cv2.getTrackbarPos("V max", "controls")

        # sauvegarder l'état avant modification
        goodHSVRange_history.append(((goodHSVRange[0].copy(), goodHSVRange[1].copy()),min_h, max_h, min_s, max_s, min_v, max_v))

        if goodHSVValue[0] != -1:

            # récupérer les valeurs HSV du point clicked
            h, s, v = int(goodHSVValue[0]), int(goodHSVValue[1]), int(goodHSVValue[2])
            
            # augmenter goodHSVRange et trackbarRange
            goodHSVRange = PushHSVLimits("good", goodHSVRange, goodHSVValue)
            min_h, max_h = increaseCyclicRange("trackbar H", (min_h, max_h), h, 180)
            min_s, max_s = increaseRange("trackbar S", (min_s, max_s), s)
            min_v, max_v = increaseRange("trackbar V", (min_v, max_v), v)
            goodHSVValue = (-1, -1, -1)  # reset to avoid continuous updates

        if badHSVValue[0] != -1:
            
            h, s, v = int(badHSVValue[0]), int(badHSVValue[1]), int(badHSVValue[2])
            # ignorer si le point est dans goodHSVRange sur les 3 composantes simultanément
            if isInCyclicRange(h, (int(goodHSVRange[0][0]), int(goodHSVRange[1][0]))) \
            and isInRange(s, (int(goodHSVRange[0][1]), int(goodHSVRange[1][1]))) \
            and isInRange(v, (int(goodHSVRange[0][2]), int(goodHSVRange[1][2]))):
                print("point is in goodHSVRange, ignoring bad click")
            else:
                # cherche les composantes dans la trackbar et réduit la moins coûteuse
                candidates = {}
                if isInCyclicRange(h, (min_h, max_h)):
                    new_min_h, new_max_h = reducecyclicRange("trackbar H", (min_h, max_h), h, 180)
                    if new_min_h <= goodHSVRange[0][0] and new_max_h >= goodHSVRange[1][0]:
                        candidates['h'] = min(CyclicDist(h, min_h, 180), CyclicDist(h, max_h, 180))

                if isInRange(s, (min_s, max_s)):
                    new_min_s, new_max_s = reduceRange("trackbar S", (min_s, max_s), s)
                    if new_min_s <= goodHSVRange[0][1] and new_max_s >= goodHSVRange[1][1]:
                        candidates['s'] = min(abs(s - min_s), abs(s - max_s))

                if isInRange(v, (min_v, max_v)):
                    new_min_v, new_max_v = reduceRange("trackbar V", (min_v, max_v), v)
                    if new_min_v <= goodHSVRange[0][2] and new_max_v >= goodHSVRange[1][2]:
                        candidates['v'] = min(abs(v - min_v), abs(v - max_v))

                if not candidates:
                    print("cannot exclude point without removing a goodHSVRange point")
                else:
                    best = min(candidates, key=candidates.get)
                    if best == 'h':
                        min_h, max_h = new_min_h, new_max_h
                    elif best == 's':
                        min_s, max_s = new_min_s, new_max_s
                    else:
                        min_v, max_v = new_min_v, new_max_v
            badHSVValue = (-1, -1, -1)

        cv2.setTrackbarPos("H min", "controls", min_h)
        cv2.setTrackbarPos("H max", "controls", max_h)
        cv2.setTrackbarPos("S min", "controls", min_s)
        cv2.setTrackbarPos("S max", "controls", max_s)
        cv2.setTrackbarPos("V min", "controls", min_v)
        cv2.setTrackbarPos("V max", "controls", max_v)

        #print
        print(f"H values: {min_h}, {goodHSVRange[0][0]}, {goodHSVRange[1][0]}, {max_h}")
        print(f"S values: {min_s}, {goodHSVRange[0][1]}, {goodHSVRange[1][1]}, {max_s}")
        print(f"V values: {min_v}, {goodHSVRange[0][2]}, {goodHSVRange[1][2]}, {max_v}")
        
        #sort them 
        print("\n")

    # decide what to do next based on key press
    key = cv2.waitKey(10)
    
        
    if key == 32:  # space key
        ret, frame = cap.read()
        j += 1
        print(f"reading next frame: {j}")
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            j = 0
            continue

    if key == 122: # 'z' key
        if goodHSVRange_history:
            goodHSVRange, min_h, max_h, min_s, max_s, min_v, max_v = goodHSVRange_history.pop()
            cv2.setTrackbarPos("H min", "controls", min_h)
            cv2.setTrackbarPos("H max", "controls", max_h)
            cv2.setTrackbarPos("S min", "controls", min_s)
            cv2.setTrackbarPos("S max", "controls", max_s)
            cv2.setTrackbarPos("V min", "controls", min_v)
            cv2.setTrackbarPos("V max", "controls", max_v)
            print(f"undo — goodHSVRange restored to: {goodHSVRange}")
        else:
            print("nothing to undo")
    
    if key == 13:  # enter key
        print(f"saved color range: lower={lower}, upper={upper}")
        cv2.destroyWindow(f"Color Detection for marker {i}")
        new_plages_HSV[i] = (lower, upper)
        i += 1
        first_time = True
        if i >= len(HSV_ranges):
            print("finished tuning all colors")
            break
        
        setTrackbarsValues(HSV_ranges[i])
        goodHSVValue = (-1, -1, -1)
        badHSVValue = (-1, -1, -1)

        h = getMedian(HSV_ranges[i][0][0],HSV_ranges[i][1][0])
        bad_h = getCyclicrangeAntimedian(HSV_ranges[i][0][0],HSV_ranges[i][1][0])
        s = getMedian(HSV_ranges[i][0][1],HSV_ranges[i][1][1])
        bad_s = getAntiMedian(HSV_ranges[i][0][1],HSV_ranges[i][1][1],255)
        v = getMedian(HSV_ranges[i][0][2],HSV_ranges[i][1][2])
        bad_v = getAntiMedian(HSV_ranges[i][0][2],HSV_ranges[i][1][2],255)

        goodHSVRange = (np.array([h, s, v]), np.array([h+1, s+1, v+1]))
        print(f"\nTesting color range: lower={HSV_ranges[i][0]}, upper={HSV_ranges[i][1]}")
        print(f"initial goodHSVRange: min = {goodHSVRange[0]}, max = {goodHSVRange[1]}")
    
    if key == 27:  # escape key
        print(f"finished tuning color range: lower={lower}, upper={upper}")
        new_plages_HSV[i] = (lower, upper)
        break

# finished tuning
cv2.destroyAllWindows()


## ======== printing results ========

print(f"\nFinal color ranges:")
for i, (lower, upper) in enumerate(new_plages_HSV):
    print(f"Marker {i}: H = {lower[0]}-{upper[0]}, S = {lower[1]}-{upper[1]}, V = {lower[2]}-{upper[2]}")
print("\n Copy-paste-ready version for the 'AllVideoFunctions' file': ")
for i, (lower, upper) in enumerate(new_plages_HSV):
    print(f"    \"{Markers_colors[i]}\": (np.array([{lower[0]}, {lower[1]}, {lower[2]}]), np.array([{upper[0]}, {upper[1]}, {upper[2]}])),")
print(" kernel size:", kernel_size)
print(" min area:", min_area)