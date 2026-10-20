"""
Rubiks cube solver using opencv and kociemba, first takes webcam input and scans the 6 faces of the cube, 
then uses kociemba and displays the cube algorithm to solve the cube 

Main controller for scanner and solver integrating 
webcam_capture.py and solver.py

Scanner starts ->
capture all six faces ->
get confirmation from the user to proceed ->
calculate solution ->
show turning guide and solution ->
at any point press e to start the process all over again or 
q to quit 

if solved cube scanned, reutrn 'solved cube, scramble and try again'


"""
