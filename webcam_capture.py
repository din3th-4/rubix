'''
webcam capture file for the solver 
contains functions for all processes related to the cam 

User is presented with a grid a bit off centre on the screen for the cube faces 
and then the program guides the user through the rotation for all the face captures
remembers the face facing the user at the end 

'''

import cv2
import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment



#camera configuration
camera_index = 0    #uses default cam 
d_frame_width = 1280  
d_frame_height = 720 

#scanning grid configuration
grid_size = 3
grid_width = 300 
grid_height = 300

#top left corner of the grid
grid_x = 750 
grid_y = 200 


show_lab_vals = True
show_sampling_regions = True 


#keyboard controls 
key_capture = ord(' ')  # spacebar: capture the current face
key_quit = ord('q')     # q: quit the program
key_retake = ord('r')    # r: retake the current face



#window configuration
window_name = "Rubik's Cube Scanner"

#cube configuration 

face_order = ['B', 'R', 'F', 'L', 'U', 'D']  # order of faces to scan
total_faces = len(face_order)

face_instructions = {
    'B' : (
        'Show the BACK face (the face facing the computer). Keep the top face pointing up.'
    ),
    'R': (
        'Turn the cube one quarter-turn to you right.'
        ' Keep the top face up.'
    ),
    'F': (
        'Turn right again. Show the original FRONT face (this was the face pointing to you when you started).'
    ),
    'L': (
        'Turn right again. Show the original LEFT face. '
    ),
    'U': (
        'Turn right once more to restore the starting position'
        ' then tilt the TOP face toward the camera.'
    ),
    'D': (
        'Restore the starting orientation, then tilt the'
        ' BOTTOM toward the camera and return to starting postion. '
    ),
}

#stores lab measurements for each captured face 
cap_faces = {face: None for face in face_order}  # dictionary to store captured face images



def initialize_camera():
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        camera.release()
        print ('Error: Could not open webcam.')
        return None 

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, d_frame_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, d_frame_height)

    return camera

#get mirrored frame 
def get_cam_frame(camera):
    # read and mirror the next cam frame 

    success, frame = camera.read()
    if not success:
        return None

    #mirror preview 
    
    return cv2.flip(frame, 1) 


def get_grid_geometry(frame):
    #calculate grid coordinates for the actual frame 

    frame_height, frame_width = frame.shape[:2]

    scale_x = frame_width / d_frame_width
    scale_y = frame_height / d_frame_height

    x = int(grid_x * scale_x)
    y = int(grid_y * scale_y)

    width = int(grid_width * scale_x)
    height = int(grid_height * scale_y)

    #make sure grid remains inside frame 
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))

    width = min(width, frame_width - x)
    height = min(height, frame_height - y)

    # Calculate the dimensions of each of the nine cells.
    cell_width = width // grid_size
    cell_height = height // grid_size

    return {
        'x': x,
        'y': y,
        'width': width,
        'height': height,
        'cell_width': cell_width,
        'cell_height': cell_height
    }


def draw_grid(frame):
    #draw the grid over the mirrored cam frame 

    grid = get_grid_geometry(frame)

    x = grid['x']
    y = grid['y']
    width = grid['width']
    height = grid['height']

    #drawin the lines 
    for i in range(grid_size + 1):

        vertical_x = x + round(i * width / grid_size)
        horizontal_y = y + round(i * height / grid_size)

        cv2.line(
            frame,
            (vertical_x, y),
            (vertical_x, y + height),
            (0, 255, 0),
            2,
        )

        cv2.line(
            frame,
            (x, horizontal_y),
            (x + width, horizontal_y),
            (0, 255, 0),
            2,
        )

    return frame




def draw_instructions(frame, message, progress = ''):
    #for displaying the instructions and scanning progress
    
    lines = message.split('\n')

    panel_height = 75 + 30 * max(0, len(lines) - 1)

    #dark background behind the instuctions
    cv2.rectangle(
        frame, 
        (15, 15),
        (min(frame.shape[1] - 15, 700), panel_height),
        (30, 30, 30),
        -1,
    )

    y_position = 42

    for line in lines:
        cv2.putText(
            frame,
            line,
            (25, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y_position += 27

    if progress:
        cv2.putText(
            frame,
            progress,
            (25, panel_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame 


def convert_to_lab(frame):
    #converting bgr to lab 

    return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

def sample_colors(lab_frame):
    """Return nine robust lab samples in row-major order.

    Each sticker's central region is split into a 3x3 set of patches. The
    median LAB value of each patch is calculated. We then find the patch
    medoid (the patch most representative of the others) and aggregate the
    closest patches. This reduces the effect of a small glare spot or a
    finger/shadow affecting only part of the sampling region.

    The return format remains a list of nine (L, A, B) tuples.
    """

    grid = get_grid_geometry(lab_frame)

    x = grid['x']
    y = grid['y']

    cell_width = grid['cell_width']
    cell_height = grid['cell_height']

    samples = []

    for row in range(grid_size):

        for col in range(grid_size):
            cell_x = x + col * cell_width
            cell_y = y + row * cell_height

            #sample the central 50%

            insert_x = cell_width // 4
            insert_y = cell_height // 4

            x1 = cell_x + insert_x
            y1 = cell_y + insert_y

            x2 = cell_x + cell_width - insert_x
            y2 = cell_y + cell_height - insert_y

            region = lab_frame[y1:y2, x1:x2]

            if region.size == 0:
                samples.append((0, 0, 0))
                continue 

            # Divide the region into nine smaller patches and measure each.
            patch_medians = []
            region_height, region_width = region.shape[:2]
            for patch_row in range(3):
                for patch_col in range(3):
                    px1 = round(patch_col * region_width / 3)
                    px2 = round((patch_col + 1) * region_width / 3)
                    py1 = round(patch_row * region_height / 3)
                    py2 = round((patch_row + 1) * region_height / 3)
                    patch = region[py1:py2, px1:px2]
                    if patch.size:
                        patch_medians.append(
                            np.median(patch.reshape(-1, 3), axis=0)
                        )

            if not patch_medians:
                samples.append((0, 0, 0))
                continue

            patch_medians = np.asarray(patch_medians, dtype=float)

            # Use the most central patch color as the reference, then keep
            # the five patch colors closest to it (or all if fewer than 5).
            # Convert patch colors to normalized LAB before comparing them.
            normalized_patches = normalize_lab(patch_medians)

            # Calculate distances between all pairs of patches.
            pairwise_distances = np.linalg.norm(
                normalized_patches[:, None, :] - normalized_patches[None, :, :],
                axis=2
            )

            # Find the patch that is most representative of the others.
            medoid_index = int(
                np.argmin(np.sum(pairwise_distances, axis=1))
            )

            distances = pairwise_distances[medoid_index]

            keep_count = min(5, len(patch_medians))
            keep_indices = np.argsort(distances)[:keep_count]

            robust_lab = np.median(patch_medians[keep_indices], axis=0)
            
            samples.append(tuple(int(round(v)) for v in robust_lab))

    return samples

def display_colors(frame, samples):
    """
    Draw a colour swatch and optional LAB values inside
    each grid cell.

    The swatch uses BGR values for display. The measurements
    themselves remain in LAB colour space.
    """

    grid = get_grid_geometry(frame)

    x = grid["x"]
    y = grid["y"]

    cell_width = grid["cell_width"]
    cell_height = grid["cell_height"]

    for index, lab_values in enumerate(samples):

        row = index // grid_size
        col = index % grid_size

        cell_x = x + col * cell_width
        cell_y = y + row * cell_height

        insert_x = cell_width // 4
        insert_y = cell_height // 4

        # Get the corresponding region from the BGR frame.
        region = frame[
            cell_y + insert_y:cell_y + cell_height - insert_y,
            cell_x + insert_x:cell_x + cell_width - insert_x,
        ]

        if region.size == 0:
            continue

        
        median_bgr = np.median(
            region.reshape(-1, 3),
            axis=0,
        ).astype(int)

        bgr_color = tuple(int(value) for value in median_bgr)

        # Draw a small colour swatch in the top-left corner.
        cv2.rectangle(
            frame,
            (cell_x + 8, cell_y + 8),
            (cell_x + 28, cell_y + 28),
            bgr_color,
            -1,
        )

        if show_lab_vals:
            label = f"L{lab_values[0]} A{lab_values[1]} B{lab_values[2]}"
            cv2.putText(
                frame, label, (cell_x + 5, cell_y + cell_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA,
            )
    
    return frame

def normalize_lab(lab_values):
    #Convert OpenCV's 8-bit LAB values to a consistent scale
    lab_values = np.asarray(lab_values, dtype=float)

    normalized = lab_values.copy()
    normalized[..., 0] = normalized[..., 0] * 100.0 / 255.0
    normalized[..., 1] = normalized[..., 1] - 128.0
    normalized[..., 2] = normalized[..., 2] - 128.0

    return normalized



def classify_face_colors(faces, max_distance=None, min_margin=8.0):
    """
    Classify stickers using the six center stickers as color references.

    Keeps all six center stickers fixed and assigns the remaining stickers
    using minimum-cost assignment, enforcing exactly nine stickers per color.

    Uncertain results include the nearest-center label and the final
    capacity-constrained label.
    """


    # Validate that all six faces were captured.
    if any(faces.get(face) is None for face in face_order):
        raise ValueError(
            'All six faces must be captured before classification.'
        )

    for face in face_order:
        measurements = np.asarray(faces[face])

        if measurements.shape != (9, 3):
            raise ValueError(
                f'Face {face} must contain exactly 9 LAB measurements.'
            )

        if not np.all(np.isfinite(measurements)):
            raise ValueError(
                f'Face {face} contains invalid LAB measurements.'
            )

    # Use each center sticker as the reference for its color.
    references = {
        face: normalize_lab(faces[face][4])
        for face in face_order
    }

    reference_labels = list(face_order)
    reference_values = np.asarray([
        references[label] for label in reference_labels
    ])

    classified = {
        face: [None] * 9
        for face in face_order
    }

    stickers = []

    # Keep center stickers fixed; collect the other 48 stickers.
    for face in face_order:
        measurements = normalize_lab(faces[face])

        for sticker_index, measurement in enumerate(measurements):

            if sticker_index == 4:
                classified[face][sticker_index] = face
                continue

            distances = np.linalg.norm(
                reference_values - measurement,
                axis=1
            )

            order = np.argsort(distances)
            best_index = int(order[0])
            second_index = int(order[1])

            stickers.append({
                'face': face,
                'index': sticker_index,
                'distances': distances,
                'nearest': reference_labels[best_index],
                'best_distance': float(distances[best_index]),
                'second_distance': float(distances[second_index]),
                'margin': float(
                    distances[second_index] - distances[best_index]
                ),
            })

    # Each color must receive exactly 8 non-center stickers.
    # Repeat each color 8 times to create 48 assignment slots.
    color_slots = [
        color
        for color in reference_labels
        for _ in range(8)
    ]

    # Cost matrix: one row per sticker, one column per color slot.
    cost_matrix = np.zeros(
        (len(stickers), len(color_slots)),
        dtype=float
    )

    for sticker_index, sticker in enumerate(stickers):
        for slot_index, color in enumerate(color_slots):
            color_index = reference_labels.index(color)
            cost_matrix[sticker_index, slot_index] = (
                sticker['distances'][color_index]
            )

    # Find the globally minimum-cost assignment.
    row_indices, column_indices = linear_sum_assignment(cost_matrix)

    for row, column in zip(row_indices, column_indices):
        sticker = stickers[row]
        assigned_color = color_slots[column]

        classified[sticker['face']][sticker['index']] = assigned_color

    # Report low-confidence stickers and labels changed by the assignment.
    uncertain = []

    for sticker in stickers:
        face = sticker['face']
        index = sticker['index']
        original_label = sticker['nearest']
        assigned_label = classified[face][index]

        is_uncertain = (
            (
                max_distance is not None
                and sticker['best_distance'] > max_distance
            )
            or sticker['margin'] < min_margin
            or original_label != assigned_label
        )

        if is_uncertain:
            uncertain.append({
                'face': face,
                'sticker': index + 1,
                'best_match': original_label,
                'assigned_match': assigned_label,
                'distance': round(sticker['best_distance'], 1),
                'margin': round(sticker['margin'], 1),
            })

    return classified, uncertain, references


def capture_face(samples, face_label):
    #store an independent copy of lab measurements 

    if face_label not in cap_faces:
        raise ValueError(
            f'Unknown face label: {face_label}'
        )

    if len(samples) != grid_size * grid_size:
        raise ValueError(
            'A face must have exactly 9 colors'
        )

    cap_faces[face_label] = np.array(
        samples,
        dtype=np.int16,
    ).copy()

    print (f'captured face : {face_label}') 

def get_next_face(face_index):
    #return instructions fro the current scanning step

    if face_index >= total_faces:
        return "All six faces captured!\nPress Q to finish."

    face_label = face_order[face_index]

    return (
        f"Face {face_index + 1}/{total_faces}: {face_label}\n"
        f"{face_instructions[face_label]}"
    )



def release_camera(camera):
    #releasing the cam

    if camera is not None:
        camera.release()

    cv2.destroyAllWindows()


def draw_sampling_regions(frame):
    # Draw rectangles around the exact regions used for LAB sampling.
    # This function only draws on the display frame.

    grid = get_grid_geometry(frame)
    x, y = grid['x'], grid['y']
    cell_width, cell_height = grid['cell_width'], grid['cell_height']

    for row in range(grid_size):
        for col in range(grid_size):
            cell_x = x + col * cell_width
            cell_y = y + row * cell_height
            inset_x = cell_width // 4
            inset_y = cell_height // 4

            cv2.rectangle(
                frame,
                (cell_x + inset_x, cell_y + inset_y),
                (cell_x + cell_width - inset_x, cell_y + cell_height - inset_y),
                (0, 255, 255),
                2,
            )
    return frame


 

def scan_cube():
    """Capture all six faces and return their LAB measurements."""
    global cap_faces
    cap_faces = {face: None for face in face_order}

    camera = initialize_camera()
    if camera is None:
        return None

    face_index = 0
    message = get_next_face(face_index)

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            frame = get_cam_frame(camera)
            if frame is None:
                print('ERROR: Could not read a camera frame')
                return None

            # Sample from the untouched frame; overlays are drawn afterward.
            lab_frame = convert_to_lab(frame)
            samples = sample_colors(lab_frame)

            frame = draw_grid(frame)
            frame = display_colors(frame, samples)
            if show_sampling_regions:
                frame = draw_sampling_regions(frame)

            progress = (
                f'Captured: {face_index}/{total_faces}'
                ' | SPACE: capture | R: retake | Q: quit'
            )
            frame = draw_instructions(frame, message, progress)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == key_quit:
                if face_index == total_faces:
                    print('All six faces captured.')
                    return {face: values.copy() for face, values in cap_faces.items()}
                print('Scanning cancelled.')
                return None

            elif key == key_capture:
                if face_index < total_faces:
                    face_label = face_order[face_index]
                    capture_face(samples, face_label)
                    face_index += 1
                    message = get_next_face(face_index)

            elif key == key_retake:
                if face_index > 0:
                    face_index -= 1
                    face_label = face_order[face_index]
                    cap_faces[face_label] = None
                    message = f'Retaking face {face_label}.\n{face_instructions[face_label]}'
                    print(f'Retaking face: {face_label}')
                else:
                    message = 'No captured face to retake yet.\n' + get_next_face(face_index)
    finally:
        release_camera(camera)


if __name__ == '__main__':
    faces = scan_cube()

    if faces is not None:
        print('\nAll captured face measurements:')
        for face, measurements in faces.items():
            print(f'\n{face}:')
            print(measurements)

        try:
            classified, uncertain, references = classify_face_colors(faces)
            print('\nCenter-sticker LAB references:')
            for face, reference in references.items():
                print(f'{face}: {np.round(reference, 1)}')

            print('\nClassified stickers (face labels are center-color labels):')
            for face, labels in classified.items():
                print(f'{face}: {labels}')

            

            all_labels = [
                label
                for labels in classified.values()
                for label in labels
            ]

            counts = Counter(all_labels)

            print('\nClassified color counts:')
            for label in face_order:
                print(f'{label}: {counts.get(label, 0)}')
            

            if uncertain:
                print('\nPotentially uncertain classifications (check these stickers):')
                for item in uncertain:
                    print(item)
            else:
                print('\nNo ambiguous matches were flagged by the margin check.')
        except ValueError as error:
            print(f'Classification skipped: {error}')




