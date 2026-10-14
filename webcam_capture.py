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
camera_index = 0
frame_width = 1280 
frame_height = 720 

#scanning grid configuration
grid_size = 3
grid_width = 300 
grid_height = 300

#top left corner of the grid
grid_x = 750 
grid_y = 200 

#size of grid cells 
cell_width = grid_width // grid_size
cell_height = grid_height // grid_size


#keyboard controls 

key_capture = ord(' ')  # spacebar: capture the current face
key_quit = ord('q')     # q: quit the program
key_retake = ord('r')    # r: retake the current face



def initalize_camera():
    pass

def draw_grid():
    pass

def draw_instructions():
    pass

def release_camera():
    pass

def sample_colors():
    pass

def convert_to_lab():
    pass

def display_colors():
    pass

def get_next_face():
    pass    

def scan_cube():
    pass

