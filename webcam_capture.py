'''
webcam capture file for the solver 
contains functions for all processes related to the cam 

User is presented with a grid a bit off centre on the screen for the cube faces 
and then the program guides the user through the rotation for all the face captures
remembers the face facing the user at the end 

'''

import cv2
import numpy as np

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

#face data storage
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
    frame = cv2.flip(frame, 1)
    return frame

#calculate grid coordinates for the actual frame 
def get_grid_geometry(frame):

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
    #samples the cnetral regiion of each grid cell 
    # 
    #returns the nine lab measurements ordered from top left
    #to bottom right, row by row 

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

            #if median is less affected by isolated reflections
            median_lab =np.median(
                region.reshape(-1, 3), axis=0
            )

            samples.append(
                tuple(int(v) for v in median_lab)
            )

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
            label = (
                f"L{lab_values[0]} "
                f"A{lab_values[1]} "
                f"B{lab_values[2]}"
            )

            cv2.putText(
                frame,
                label,
                (cell_x + 5, cell_y + cell_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return frame


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


 

def scan_cube():
    '''
    scan all 6 faces and returns their lab measurements

    space: capture the current face
    r: retake the most recently captured face
    q: quit ; return data only if all 6 faces are captured 

    '''

    global cap_faces

    #reset storage when a new scan begins 
    cap_faces = {
        face: None for face in face_order 
    }

    camera = initialize_camera()

    if camera is None:
        return None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    face_index = 0
    message = get_next_face(face_index)

    try:

        while True:

            frame = get_cam_frame(camera)

            if frame is None:
                print ('ERROR: Could not read a camera frame')
                return None 

            #convert current fraame to lab
            lab_frame = convert_to_lab(frame)
            samples = sample_colors(lab_frame)

            #draw interface on mirrored preview.
            frame = draw_grid(frame)
            frame = display_colors(frame, samples)

            progress = (
                f'Captured: {face_index}/{total_faces}'
                ' | SPACE: capture | r: retake | q: quit'
            )

            frame = draw_instructions(
                frame,
                message,
                progress,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF


            #quit 


            if key == key_quit:

                if face_index == total_faces:
                    print ('All six faces captured.')

                    return {
                        face: values.copy() 
                        for face, values in cap_faces.items()
                    }

                print ('scanning cancelled.')
                return None 

            #capture current face 
            elif key == key_capture:

                if face_index < total_faces:

                    face_label = face_order[face_index]

                    capture_face(samples, face_label)

                    face_index += 1
                    message = get_next_face(face_index)

            #retake previously captured face 
            elif key == key_retake:

                if face_index > 0:

                    face_index -= 1

                    face_label = face_order[face_index]

                    cap_faces[face_label] = None

                    message = (
                        f"Retaking face {face_label}.\n"
                        f"{face_instructions[face_label]}"
                    )

                    print(f"Retaking face: {face_label}")

                else:
                    message = (
                        "No captured face to retake yet.\n"
                        + get_next_face(face_index)
                    )

    finally: 
        release_camera(camera)



#for testing independently 

if __name__ == "__main__":

    faces = scan_cube()

    if faces is not None:
        print("\nAll captured face measurements:")

        for face, samples in faces.items():
            print(f"\n{face}:")
            print(samples)


            
