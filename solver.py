'''
python file for integrating the data from the webcam_capture file with 
kociemba

first the file converts the face data into the format expected by kociemba 

'''

import numpy as np 

import kociemba

from collections import Counter 

#kociembas required face order 
k_face_order = ['U', 'R', 'F', 'D', 'L', 'B']
valid_face_labels = set(k_face_order)


def validate_classified_faces(faces):
    '''
    validates the classified sticker labels returned by
    webcam_capture.classify_face_colors()

    '''

    if not isinstance (faces, dict):
        raise TypeError('Faces must be in dictionary')

    if set(faces.keys()) != valid_face_labels:
        raise ValueError(
            "Expected exactly the faces U, R, F, D, L and B."
        )

    for face in k_face_order:
        if len(faces[face]) != 9:
             raise ValueError(
                f"Face {face} must contain exactly 9 stickers."
            )

        if any(label not in valid_face_labels for label in faces[face]):
            raise ValueError(
                f"Face {face} contains an invalid sticker label."
            )

    counts = Counter(
        label 
        for face in k_face_order
        for label in faces[face]

    )

    for label in k_face_order:
        if counts[label] != 9: 
            raise ValueError (
                f'Expected 9 stickers labelled {label}, '
                f'but found {counts[label]}'
            )

    return True 



def orient_faces(faces):
    '''
    Reorient the six captured face grids into kociemba's required 
    face orientations.

    (webcam captures are mirrored)

    capture order: B, R, F, L, U, D
    kociemba's facelet ordering: U, R, F, D, L, B


    B, R, F, L: mirrored webcam preview should be corrected
    U: rows should be reversed cuz top face is titled toward 
    the cam from the back side 
    D: sampled orientation is kept unchanged 

    '''

    validate_classified_faces(faces)

    oriented = {}

    for face in k_face_order:

        #convert the 9 stickers into a 3x3 grid 
        grid = [
            list(faces[face][0:3]),
            list(faces[face][3:6]),
            list(faces[face][6:9])
        ]

        if face in ('B', 'R', 'F', 'L'):
            #correcting the mirrored webcam preview 
            grid = [row[::-1] for row in grid]

        elif face == 'U':
            #reverse the row order for the bottom face
            grid = grid[::-1]

        elif face == 'D':
            # Reverse rows for the bottom face
            grid = grid[::-1]


        #convert the corrected grid back to the 9 stickers 
        oriented[face] = [
            sticker 
            for row in grid 
            for sticker in row 
        ]

    return oriented 



def convert_to_facelet_string(faces):
    """
    Convert classified and oriented face data into the
    54-character facelet string expected by Kociemba.
    """

    oriented_faces = orient_faces(faces)

    facelet_string = ''.join(
        ''.join(oriented_faces[face])
        for face in k_face_order
    )
    validate_facelet_string(facelet_string)

    return facelet_string


def validate_facelet_string(facelet_string):
    '''
    Valdate the format of the kociemba string 
    '''

    if not isinstance(facelet_string, str):
        raise TypeError(
            f'Facelet string must contain 54 characters, '
            f'but contains {len(facelet_string)}'
        )


    invalid_labels = set(facelet_string) - valid_face_labels

    if invalid_labels:
        raise ValueError(
            f'Invalid face lables found: {invalid_labels}'
        )

    counts = Counter(facelet_string)

    for face in k_face_order:
        if counts[face] != 9:
            raise ValueError(
                f"Expected 9 stickers labelled {face}, "
                f"but found {counts[face]}."
            )

    #each face's centre sticker must match its face label 
    for face_index, face in enumerate(k_face_order):
        centre_index = face_index * 9 + 4 

        if facelet_string[centre_index] != face:
            raise ValueError(
                f'Centre sticker for face {face} is incorrect'
            )


    return True 

    


def solve_cube(facelet_string):
    '''
    Solving the cube using kociemba 

    returns the solution as a string of moves 

    '''

    validate_facelet_string(facelet_string)

    try: 
        solution = kociemba.solve(facelet_string)

    except Exception as error:
        print("Actual Kociemba error:", repr(error))
        raise

    return solution



if __name__ == "__main__":
    from webcam_capture import scan_cube, classify_face_colors

    print("Starting Rubik's Cube scan...")

    raw_faces = scan_cube()

    if raw_faces is None:
        print("Scan cancelled or camera unavailable.")

    else:
        try:
            print("\nClassifying stickers...")
            classified_faces, uncertain, references = (
                classify_face_colors(raw_faces)
            )
            print("\nClassified faces:")
            for face in ['B', 'R', 'F', 'L', 'U', 'D']:
                print(f"{face}: {classified_faces[face]}")

            if uncertain:
                print(f"\nWarning: {len(uncertain)} uncertain stickers.")
                for sticker in uncertain:
                    print(sticker)

            print("\nConverting faces to Kociemba format...")
            facelet_string = convert_to_facelet_string(
                classified_faces
            )

            #temp debug: inspect the oriented face grids
            oriented = orient_faces(classified_faces)

            print("\nOriented face grids:")
            for face in k_face_order:
                print(f"\n{face}:")
                for i in range(0, 9, 3):
                    print(oriented[face][i:i+3])


            print("\nFacelet string:")
            print(facelet_string)
            print(f"Length: {len(facelet_string)}")

            print("\nSolving cube...")
            solution = solve_cube(facelet_string)

            print("\nSolution:")
            print(solution)

        except (ValueError, TypeError) as error:
            print(f"\nError: {error}")

