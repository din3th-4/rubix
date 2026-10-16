'''
python file for integrating the data from the webcam_capture file with 
kociemba

first the file converts the face data into the format expected by kociemba 

'''

import numpy as np 

import kociemba

from webcam_capture import scan_cube, classify_face_colors

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

    kociemba's facelet ordering: U, R, F, D, L, B


    The webcam preview is mirrored horizontally 
    side faces and U are corrected horizontally 
    D is corrected vertically because it is viewed from below 
    while being tilted towards the cam

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

        if face in ('B', 'R', 'F', 'L', 'U'):
            #correcting the mirrored webcam preview 
            grid = [row[::-1] for row in grid]

        elif face == 'D':
            #reverse the row order for the bottom face
            grid == grid[::-1]


        #convert the corrected grid back to the 9 stickers 
        oriented[face] = [
            sticker 
            for row in grid 
            for sticker in row 
        ]

    return oriented 





    

def convert_to_facelet_string(faces):
    pass


def validate_facelet_string(facelet_string):
    pass


def solve_cube(facelet_string):
    pass 





