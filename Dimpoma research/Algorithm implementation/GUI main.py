import HMM

import PySimpleGUI as sg

import copy
import random
import time
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import cv2

sg.theme("LightGreen")
font = ("Serif", 18)

sg.set_options(tooltip_font=("Serif 16"))
x0_tooltip = "You may choose: \
    \n* free space : type in your own preferred initial state \
    \n* random : x0 will be randomly generated according to initial uniform distribution"
implicit_I_tooltip = "You may choose: \
    \n* none : parameter won't be considered in the model \
    \n* free space : type in square brackets your own preferred integer value(s)"
q_tooltip = "You may choose: \
    \n* none : parameter won't be considered in the model \
    \n* free space : type in square brackets your own preferred float value(s)"
p0_tooltip = "You may choose: \
    \n* free space : type in your own preferred float value \
    \n* random : p0 will be randomly generated from the interval (0.05, 0.95)"
q0_tooltip = "You may choose: \
    \n* none : parameter won't be considered in the model \
    \n* free space : type in square brackets your own preferred float value(s) \
    \n* random : q0 will be randomly generated from the interval (0.05, 0.95)"
stop_criterion_tooltip = "You may choose: \
    \n* x iterations : type instead of x your own preferred integer value \
    \n* increments : algorithm will be stoped after reaching threshold values of P(Y=y) increments"
decoding_mode_tooltip = "You may choose: \
    \n* once only : run algorithm for the estimation with maximum P(Y=y) value among restarts  \
    \n* for each restart : run algorithm for each restart to make a statistical analysis"

initialization_column = [
    [
        sg.Frame(
            layout=[
                [
                    sg.Text("Dimention of a state vector", size=(25, 1)), 
                    sg.Text("N =", size=(4, 1)), 
                    sg.InputText(size=(11, 1), default_text="5", key="-N-"),
                ],
                [
                    sg.Text("First initiate state", size=(25, 1)), 
                    sg.Text("x0 =", size=(4, 1)), 
                    sg.Combo(
                        ["","random"],
                        default_value="random",
                        readonly=False,
                        tooltip=x0_tooltip,
                        size=(10, 1),
                        key="-X0-"
                    )
                ],
                [
                    sg.Text("Transition probability", size=(25, 1)), 
                    sg.Text("p =", size=(4, 1)), 
                    sg.InputText(size=(11, 1), default_text="0.2", key="-P-"),
                ],
                [
                    sg.Text("Length of a chain", size=(25, 1)), 
                    sg.Text("T =", size=(4, 1)), 
                    sg.InputText(size=(11, 1), default_text="200", key="-T-"),
                ],
            ],
            title="Hidden chain generation",
            font=("Serif", 18, "bold"),
            size=(640,200),
            relief=sg.RELIEF_GROOVE,
        )
    ],    
    [
        sg.Button("VIEW STATE SHAPE", size=(41, 1), key="-VIEW STATE SHAPE-"),
    ],
    [
        sg.Frame(
            layout=[
                [
                    sg.Text("Observed indices", size=(21, 1)), 
                    sg.Text("I =", size=(3, 1)), 
                    sg.InputText(size=(16, 1), default_text="[[1,2],[0,3]]", key="-OBSERVED INDEXES-"),
                ], 
                [
                    sg.Text("Implicit indices", size=(21, 1)), 
                    sg.Text("I* =", size=(3, 1)), 
                    sg.Combo(
                        ["none","[0,2,4]"],
                        default_value="[0,2,4]",
                        readonly=False,
                        tooltip=implicit_I_tooltip,
                        size=(16, 1),
                        key="-IMPLICIT INDEXES-"
                    ),                
                ],
                [
                    sg.Text("Distortion coefficients", size=(21, 1)), 
                    sg.Text("q =", size=(3, 1)), 
                    sg.Combo(
                        ["none","[0.05, 0.1]"],
                        default_value="none",
                        readonly=False,
                        tooltip=q_tooltip,
                        size=(16, 1),
                        key="-DISTORTION COEFFICIENTS-"
                    ),
                ]
            ],
            title="Observed chain generation",
            font=("Serif", 18, "bold"),
            size=(640,165),
            relief=sg.RELIEF_GROOVE,
        )
    ],
    [
        sg.Button("VIEW OBSERVED STATE SHAPE", size=(41, 1), key="-VIEW OBSERVED STATE SHAPE-"),
    ],
    [
        sg.Image(filename="Images/empty_graph.png", key="-GRAPH-")
    ],
]

algorithms_column = [
    [
        sg.Frame(
            layout=[
                [
                    sg.Text("Initial approximation", size=(31, 1)), 
                    sg.Text("p0 =", size=(4, 1)),
                    sg.Combo(
                        ["0.55","random"],
                        default_value="random",
                        readonly=False,
                        tooltip=p0_tooltip,
                        size=(11, 1),
                        key="-P0-"
                    )
                ],
                [
                    sg.Text("", size=(31, 1)), 
                    sg.Text("q0 =", size=(4, 1)),
                    sg.Combo(
                        ["none","[0.3, 0.4]","random"],
                        default_value="none",
                        readonly=False,
                        tooltip=q0_tooltip,
                        size=(11, 1),
                        key="-Q0-"
                    )
                ],
                [
                    sg.Text("Learning algorithm stop criterion", size=(36, 1)), 
                    sg.Combo(
                        ["20 iterations","increments"],
                        default_value="increments",
                        readonly=False,
                        tooltip=stop_criterion_tooltip,
                        size=(11, 1),
                        key="-STOP CRITERION-"
                    )
                ],
                [
                    sg.Text("Scale forward & backward coefficients", size=(36, 1)), 
                    sg.Combo(
                        ["true", "false"],
                        default_value="false",
                        readonly=True,
                        size=(11, 1),
                        key="-SCALING-"
                    )
                ],
                [
                    sg.Text("")
                ],
                [
                    sg.Text("Number of restarts", size=(32, 1)), 
                    sg.Text("r =", size=(3, 1)), 
                    sg.InputText(size=(12, 1), default_text="1", key="-RESTARTS-"),
                ],

            ],
            title="Learning algorithm",
            font=("Serif", 18, "bold"),
            size=(748,275),
            relief=sg.RELIEF_GROOVE,
        )
    ],
    [
        sg.Button("RUN LEARNING ALGORITHM", size=(48, 1), key="-LEARNING ALGORITHM-"),
    ],
    [
        sg.Text("")
    ],
    [
        sg.Frame(
            layout=[
                [
                    sg.Text("Restart №0", size=(20,1), key="-RESTART BAR-"),
                ],
                [
                    sg.Text("0%", size=(5,1), key="-PROGRESS BAR PERCENT-"),
                    sg.ProgressBar(
                        100, 
                        orientation="h", 
                        size=(60,15), 
                        border_width=2, 
                        bar_color=("gray","white"), 
                        key="-PROGRESS BAR-"),
                ],
                [
                    sg.Text("")
                ],
                [
                    sg.Text("Time of execution", size=(31, 1)), 
                    sg.Text("t =", size=(11, 1), key="-LEARNING ALGORITHM: TIME-"), 
                ],

            ],
            title="Progress bar",
            font=("Serif", 18, "bold"),
            size=(748,195),
            relief=sg.RELIEF_GROOVE,
        )
    ],
    [
        sg.Button("VIEW LEARNING RESULTS", size=(48, 1), key="-VIEW LEARNING RESULTS-"),
    ],
    [
        sg.Text("")
    ],
    [
        sg.Frame(
            layout=[
                [
                    sg.Text("Decoding algorithm mode", size=(30, 1)), 
                    sg.Combo(
                        ["run only once", "run for each restart"],
                        default_value="run for each restart",
                        readonly=True,
                        tooltip=decoding_mode_tooltip,
                        size=(17, 1),
                        key="-DECODING MODE-"
                    )
                ],
            ],
            title="Decoding algorithm",
            font=("Serif", 18, "bold"),
            size=(748,80), # 280
            relief=sg.RELIEF_GROOVE,
        )
    ],
    [
        sg.Button("RUN DECODING ALGORITHM & VIEW RESULTS", size=(48, 1), key="-RUN DECODING ALGORITHM & VIEW RESULTS-"),
    ],
    [
        sg.Text("")
    ],
    [
        sg.Frame(
            layout=[
                [
                    sg.Text("Estimated implicit indexes", size=(30, 1)), 
                    sg.Text("I* =", size=(17, 1), key="-ESTIMATED IMPLICIT INDEXES-"), 
                ],
            ],
            title="Estimation of implicit indexes",
            font=("Serif", 18, "bold"),
            size=(748,80),
            relief=sg.RELIEF_GROOVE,
        )
    ],
    [
        sg.Button("RUN ESTIMATION & VIEW RESULTS", size=(48, 1), key="-ESTIMATE IMPLICIT INDEXES-"),
    ],
]

layout = [
    [
        sg.Column(initialization_column),
        sg.VSeperator(),
        sg.Column(algorithms_column, vertical_alignment="top"),
    ]
]

window = sg.Window("Hidden Markov Model", layout, font=font)

learning_algorithm_indicator = False # learning algorithm hasn't done yet
decoding_algorithm_indicator = False # decoding algorithm hasn't done yet

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    elif event == "-VIEW STATE SHAPE-":
        try:
            if values["-N-"] != "":
                HMM.display_state(int(values["-N-"]))
                window["-GRAPH-"].update(filename="Images/just_states.png")
            else:
                sg.Popup("\nInput value(s)!\n", title="Error window", custom_text="Close", image="Images/oops.png", button_color="red", font=font)
        except Exception as error:
            sg.Popup(
                f"\n{error}\n",
                title="Error window", 
                image="Images/error.png",
                custom_text="Close",
                button_color="red", 
                font=("Arial", 16)
            )
        
    elif event == "-VIEW OBSERVED STATE SHAPE-":
        try:
            if (values["-N-"] and values["-OBSERVED INDEXES-"]) not in [""]:
                HMM.display_graph(eval(values["-OBSERVED INDEXES-"]), values["-IMPLICIT INDEXES-"], int(values["-N-"]))
                window["-GRAPH-"].update(filename="Images/graph.png")
            else:
                sg.Popup("\nInput value(s)!\n", title="Error window", custom_text="Close", image="Images/oops.png", button_color="red", font=font)
        except Exception as error:
            sg.Popup(
                f"\n{error}\n",
                title="Error window", 
                image="Images/error.png",
                custom_text="Close",
                button_color="red", 
                font=("Arial", 16)
            )

    elif event == "-LEARNING ALGORITHM-":
        try:
            window["-LEARNING ALGORITHM: TIME-"].update("t =")
            window["-ESTIMATED IMPLICIT INDEXES-"].update("I* =")

            N,x0,p,T,I,q,p0,q0,criterion,scaling,r = values["-N-"], \
                                                     values["-X0-"], \
                                                     values["-P-"], \
                                                     values["-T-"], \
                                                     values["-OBSERVED INDEXES-"], \
                                                     values["-DISTORTION COEFFICIENTS-"], \
                                                     values["-P0-"], \
                                                     values["-Q0-"], \
                                                     values["-STOP CRITERION-"], \
                                                     values["-SCALING-"], \
                                                     values["-RESTARTS-"]
            
            # all() method evaluates like a series of "and" operators between each of the elements
            if all([N,x0,p,T,I,q,p0,q0,criterion,scaling,r][i] != "" for i in range(len(values) - 2)):
                
                # ----------------------------- initialization ----------------------------

                N,p,T,I,r = int(N), float(p), int(T), eval(I), int(r)

                if x0 == "random":
                    initial_state_decimal = [random.randint(0,pow(2,N)-1) for i in range(r)]
                    initial_state = [HMM.binary(initial_state_decimal[i],N) for i in range(r)]
                else:
                    initial_state = [x0 for i in range(r)]

                if p0 == "random":
                    initial_p_approximation = [random.uniform(0.05,0.95) for i in range(r)]
                else:
                    initial_p_approximation = [float(p0) for i in range(r)]

                if values["-DISTORTION COEFFICIENTS-"] == "none" and values["-Q0-"] == "none":
                    initial_q_approximation = ["none" for i in range(r)]
                    estimator = "parameter p estimation task (distortion-free model)"
                elif values["-DISTORTION COEFFICIENTS-"] != "none" and values["-Q0-"] == "none":
                    initial_q_approximation = [eval(values["-DISTORTION COEFFICIENTS-"]) for i in range(r)]
                    estimator = "parameter p estimation task (model with distortion)"
                elif values["-DISTORTION COEFFICIENTS-"] != "none" and values["-Q0-"] != "none":
                    estimator = "parameter p and coefficients q estimation task"
                    if values["-Q0-"] == "random":
                        initial_q_approximation = [[random.uniform(0.05,0.5) for j in range(len(I))] for i in range(r)]
                    else:
                        initial_q_approximation = [eval(values["-Q0-"]) for i in range(r)]

                initial_approximation = [[initial_p_approximation[i], initial_q_approximation[i]] for i in range(r)]

                start_time = time.time()

                # arrays to keep results after all restarts
                estimated_parameters, MLE = [], []
                list_of_x_real, list_of_y = [], []

                p_statistical_estimation = []

                for k in range(r):
                    window["-RESTART BAR-"].update(f"Restart №{k+1}")

                    # ----------------------------- learning algorithm -----------------------------

                    x0 = initial_state[k]
                    x_real = HMM.generate_hidden_chain(x0,p,N,T)
                    # x = ['00010', '00010', '00011', '00111', '00111', '00111', '10111', '10101', '10111', '10111', '10101', '10001', '10101', '00101', '10101', '00101', '00101', '00001', '00101', '00111', '10111', '10101', '11101', '10101', '11101', '11111', '10111', '10110', '10110', '10100', '10101', '10111', '10011', '10010', '10010', '00010', '00110', '10110', '10100', '10110', '00110', '00100', '00110', '00110', '00100', '00000', '10000', '10010', '10011', '10010', '10000', '10001', '10000', '10001', '10011', '10001', '10001', '00001', '10001', '11001', '11101', '11001', '11101', '11101', '10101', '00101', '00001', '00001', '01001', '01001', '11001', '01001', '01001', '11001', '11001', '11011', '11111', '01111', '01110', '00110', '00110', '01110', '11110', '11111', '11111', '11011', '11011', '10011', '10010', '10110', '10010', '11010', '11000', '11001', '01001', '01011', '01011', '01011', '01001', '01101', '01111', '01101', '01111', '01110', '01110', '01111', '01011', '01001', '11001', '10001', '10101', '10101', '10100', '11100', '11000', '11000', '11100', '11101', '11101', '11100', '11000', '01000', '01001', '11001', '11011', '11011', '11011', '11001', '01001', '01011', '00011', '01011', '00011', '01011', '01001', '01011', '01001', '01011', '00011', '00010', '00110', '00110', '01110', '01110', '11110', '10110', '10110', '00110', '00010', '00011', '00001', '00101', '00001', '00011', '00011', '00111', '00011', '01011', '00011', '10011', '00011', '01011', '01001', '01000', '01001', '01011', '01011', '01001', '01001', '00001', '00000', '00010', '00010', '00000', '00100', '01100', '11100', '11101', '01101', '01100', '11100', '11101', '11001', '11001', '11001', '11001', '11001', '11000', '11000', '11100', '11101', '11101', '10101', '11101', '01101', '01100', '01100', '01101', '01100', '01110', '11110', '11010', '11011', '11111', '11110', '10110', '11110', '11110', '11010', '10010', '10110', '10100', '10110', '00110', '00111', '00110', '10110', '11110', '11110', '11111', '11111', '11111', '01111', '01011', '01011', '01011', '00011', '00111', '10111', '10101', '00101', '00101', '00100', '00000', '00000', '00100', '00101', '00100', '00101', '00001', '01001', '01101', '11101', '11111', '01111', '00111', '01111', '11111', '11011', '10011', '10111', '10111', '00111', '10111', '10101', '11101', '11100', '11100', '01100', '00100', '00110', '00111', '00111', '10111', '00111', '00111', '00110', '10110', '10110', '11110', '11111', '11111', '11011', '11011', '11001', '01001', '01011', '00011', '01011', '01011', '11011', '11111', '01111', '00111', '10111', '10110', '10111', '10011', '10111', '10011', '10111', '10110', '10010', '10010', '10010', '10010', '10110', '10110', '10110', '10010', '00010', '00010', '00000', '00100', '00000', '00001', '01001', '01101', '01101', '01100', '01101', '01001', '01011', '00011', '00010', '00011', '00001', '00001', '00011', '01011', '11011', '10011', '00011', '00001', '00001', '00011', '10011', '10001', '00001', '00000', '01000', '01000', '01010', '01110', '01100', '01101', '00101', '00111', '10111', '10110', '10100', '10100', '11100', '11000', '01000', '11000', '11001', '01001', '01001', '01101', '11101', '10101', '10001', '11001', '11001', '11101', '11001', '10001', '11001', '01001', '11001', '10001', '11001', '11001', '11011', '11111', '01111', '01111', '01101', '11101', '11101', '11111', '10111', '10101', '10001', '10001', '10001', '00001', '00101', '00101', '00100', '00100', '01100', '00100', '10100', '10101', '10100', '10110', '00110', '00100', '00101', '10101', '10001', '11001', '10001', '10001', '10011', '10011', '10001', '11001', '11101', '10101', '10111', '10110', '10100', '10101', '00101', '00100', '10100', '10000', '00000', '00100', '00100', '10100', '10110', '10110', '10010', '11010', '11000', '11010', '11110', '01110', '01010', '00010', '00011', '00111', '00101', '01101', '01101', '01101', '01101', '01101', '11101', '11100', '11100', '11101', '11001', '11000', '11001', '11000', '11010', '11000', '10000', '00000', '10000', '10010', '11010', '10010', '10000', '10000', '10010', '00010', '01010', '01010', '01011', '01010', '01011', '01011', '00011', '00011', '00111', '00101', '01101', '01111', '01111', '01111', '01111', '11111', '01111', '00111', '00101', '00001', '00011', '00011', '00001', '00011', '00111', '01111', '00111', '10111', '10110', '10110', '10110', '00110', '00110', '00010', '00000', '01000', '01010', '11010', '01010', '00010', '00010', '00000', '10000', '10001', '10101', '10001', '00001', '00101', '01101', '01111', '11111', '11111', '11101', '01101', '01111', '01011', '00011', '00001', '00001', '00011', '00001', '10001', '10000', '11000', '01000', '01001', '00001', '10001', '10000', '10100', '10000', '10010', '10010', '10010', '10010', '10010', '11010', '11010', '11010', '11110', '01110', '01111', '00111', '01111', '01111', '00111', '00111', '10111', '10101', '10100', '10000', '10000', '10000', '10100', '10110', '11110', '11110', '11100', '11100', '01100', '01101', '01100', '01100', '01100', '01000', '01000', '01100', '11100', '11110', '11010', '10010', '10010', '10011', '10011', '10010', '10010', '11010', '11110', '01110', '11110', '11110', '11110', '11111', '10111', '00111', '00011', '00010', '01010', '01110', '00110', '00100', '00110', '01110', '11110', '11110', '01110', '01110', '01100', '01110', '01110', '01100', '11100', '11100', '11101', '11100', '11110', '11100', '11110', '11110', '11010', '11110', '01110', '01100', '01101', '01001', '11001', '11011', '10011', '00011', '00010', '00110', '00010', '00110', '00111', '00110', '00100', '00110', '00111', '01111', '00111', '01111', '01011', '11011', '11010', '11010', '11000', '11000', '10000', '11000', '01000', '01000', '00000', '00000', '00100', '00101', '01101', '01101', '01100', '11100', '11110', '11010', '11010', '11010', '10010', '00010', '00000', '00100', '00000', '01000', '11000', '01000', '01000', '01100', '00100', '00000', '01000', '11000', '11001', '11000', '10000', '10100', '10100', '10100', '11100', '01100', '01110', '01110', '01111', '11111', '11011', '11001', '10001', '10001', '10011', '10011', '10001', '10011', '00011', '00011', '00011', '00010', '00010', '00000', '01000', '01100', '01101', '01100', '01101', '00101', '01101', '01101', '01100', '11100', '11000', '11001', '10001', '10001', '10000', '10010', '10000', '00000', '00001', '00001', '00011', '10011', '10011', '10010', '10110', '11110', '11110', '11010', '11011', '11111', '11110', '11111', '11011', '11011', '11010', '10010', '00010', '01010', '01010', '11010', '11110', '11111', '01111', '01111', '01101', '01101', '00101', '00001', '00000', '10000', '10000', '11000', '11010', '11010', '11000', '01000', '01100', '01101', '01001', '01000', '01000', '01001', '00001', '00001', '00101', '10101', '10101', '10111', '00111', '01111', '01011', '01010', '01000', '01000', '01000', '11000', '11100', '11000', '10000', '00000', '00000', '00100', '00110', '10110', '00110', '00110', '00110', '00010', '00000', '00100', '00100', '00100', '00000', '00000', '00001', '00001', '00101', '00100', '00101', '00101', '00111', '00110', '00111', '01111', '01101', '01101', '01101', '00101', '00101', '00101', '00100', '00101', '00100', '00101', '00001', '10001', '10011', '11011', '11011', '11111', '10111', '11111', '10111', '10110', '00110', '00111', '00110', '00010', '10010', '10000', '10010', '00010', '00010', '00110', '01110', '01111', '01111', '11111', '10111', '00111', '00011', '01011', '01111', '01011', '01001', '00001', '00000', '00100', '00101', '00101', '00111', '10111', '10110', '10110', '11110', '11100', '11100', '11110', '10110', '10010', '10011', '10001', '00001', '01001', '11001', '11101', '11100', '11000', '11000', '11000', '11001', '11001', '11011', '01011', '01010', '01000', '01100', '01110', '00110', '00110', '00110', '00111', '00111', '10111', '00111', '00101', '00100', '00110', '00110', '00111', '10111', '00111', '00101', '10101', '10100', '10110', '10110', '11110', '01110', '00110', '10110', '10110', '11110', '10110', '10100', '10100', '10101', '10001', '11001', '11011', '01011', '00011', '10011', '10010', '10110', '10100', '10100', '10101', '10101', '10111', '10011', '10010', '00010', '01010', '01110', '01111', '01011', '01011', '01001', '01000', '00000', '00001', '00011', '10011', '10010', '00010', '01010', '01011', '01010', '01011', '01111', '01110', '01111', '01110', '01010', '01010', '01011', '11011', '11011', '10011', '10010', '10010', '10010', '11010', '11010', '11011', '11011', '11001', '01001', '01001', '01011', '00011', '01011', '01111', '00111', '10111', '11111', '11110', '01110', '01111', '01011', '01011', '00011', '00001', '00101', '00001', '10001', '00001', '00001', '00011', '00001', '00000', '00100', '00100', '00100', '00110', '00100', '10100', '00100', '00100', '01100', '01101', '01001', '01000', '01100', '00100', '00101', '10101', '10111', '00111', '00101', '10101', '10111', '10111', '10111', '10011', '10010', '10000', '11000', '01000', '01100', '01100', '01000', '11000', '10000']
                    # x_real = x[:T]

                    if values["-DISTORTION COEFFICIENTS-"] == "none":
                        y = HMM.collect_observations(x_real,I,T,enumerate=True)
                    else:
                        x_distorted = HMM.distort_hidden_chain(x_real,T,eval(values["-DISTORTION COEFFICIENTS-"]),I)
                        y = HMM.collect_observations(x_distorted,I,T,enumerate=True)

                    list_of_x_real.append(copy.deepcopy(x_real))
                    list_of_y.append(copy.deepcopy(y))

                    p0, q0 = initial_approximation[k][0], initial_approximation[k][1]
                    estimated_parameters_k, joint_probabilities, joint_probabilities_increments = HMM.learning_algorithm(y,N,T,I,estimator,p0,q0,criterion,scaling,window)

                    estimated_parameters.append(copy.deepcopy(estimated_parameters_k[-1]))

                    MLE.append(copy.deepcopy(joint_probabilities[-1]))

                    p_statistical_estimation_k = HMM.statistical_p_estimation(y,N,T,I,estimator)
                    p_statistical_estimation.append(copy.deepcopy(p_statistical_estimation_k)) 

                end_time = time.time()

                # print("\nn p")
                # for i in range(len(estimated_parameters_k)):
                #     print(i, estimated_parameters_k[i][0])

                # print("\nn p q1 q2")
                # for i in range(len(estimated_parameters_k)):
                #     print(i, estimated_parameters_k[i][0], estimated_parameters_k[i][1][0], estimated_parameters_k[i][1][1])

                # print(f"\nr p")
                # for i in range(len(estimated_parameters)):
                #     print(i, estimated_parameters[i][0], p_statistical_estimation[i])
                
                # print(f"argmin to st  : {np.argmin([abs(p_statistical_estimation[i] - estimated_parameters[i][0]) for i in range(len(estimated_parameters))])}")
                # print(f"argmin to 0.2 : {np.argmin([abs(0.2 - estimated_parameters[i][0]) for i in range(len(estimated_parameters))])}")

                # print("\nr p q1 q2")
                # for i in range(len(estimated_parameters)):
                #     print(i, estimated_parameters[i][0], estimated_parameters[i][1][0], estimated_parameters[i][1][1])

                # print("\np0  :", [initial_approximation[i][0] for i in range(r)])
                # print("p*  :", estimated_parameters)
                # print("MLE :", MLE)
                # print("dp  :", [abs(estimated_parameters[i][0] - p_statistical_estimation[i]) for i in range(r)])

                if end_time - start_time < 60:
                    window["-LEARNING ALGORITHM: TIME-"].update(f"t = {round(end_time-start_time,3)} s")
                elif end_time - start_time <= 60*60:
                    window["-LEARNING ALGORITHM: TIME-"].update(f"t = {round((end_time-start_time)/60,1)} min")
                elif end_time - start_time > 60*60:
                    window["-LEARNING ALGORITHM: TIME-"].update(f"t = {round((end_time-start_time)/3600,1)} h")

                learning_algorithm_indicator = True # learning algorithm is done
            else:
                sg.Popup("\nInput value(s)!\n", title="Error window", custom_text="Close", image="Images/oops.png", button_color="red", font=font)

        except Exception as error:
            sg.Popup(
                f"\n{error}\n",
                title="Error window", 
                image="Images/error.png",
                custom_text="Close",
                button_color="red", 
                font=("Arial", 16)
            )

    elif event == "-VIEW LEARNING RESULTS-":
        try:        
            if learning_algorithm_indicator == True: 
                if r == 1:
                    HMM.display_convergence(estimated_parameters_k, joint_probabilities, p_statistical_estimation_k)
                else:                     
                    HMM.display_estimated_parameters(estimated_parameters,MLE,p_statistical_estimation)
            else:
                sg.Popup("\nRun learning algorithm!\n", title="Error window", custom_text="Close", image="Images/oops.png", button_color="red", font=font)
        except Exception as error:
            sg.Popup(
                f"\n{error}\n",
                title="Error window", 
                image="Images/error.png",
                custom_text="Close",
                button_color="red", 
                font=("Arial", 16)
            )

    elif event == "-RUN DECODING ALGORITHM & VIEW RESULTS-":
        try:
            if learning_algorithm_indicator == True:
                window["-ESTIMATED IMPLICIT INDEXES-"].update("I* =")

                start_time = time.time()

                x_hamming_distances, x_mismatch_indexes = [], []
                y_hamming_distances, y_mismatch_indexes = [], []

                # ---------------------- decoding algorithm ----------------------

                list_of_x_predicted = []

                if values["-DECODING MODE-"] == "run only once":

                    x_predicted, x_hamming_distances_k, x_mismatch_indexes_k, y_hamming_distances_k, y_mismatch_indexes_k = HMM.viterbi_algorithm(list_of_x_real[np.argmax(MLE)],list_of_y[np.argmax(MLE)],N,T,I,estimated_parameters[np.argmax(MLE)],estimator,values)

                    x_hamming_distances.append(x_hamming_distances_k)
                    x_mismatch_indexes.append(x_mismatch_indexes_k)

                    y_hamming_distances.append(y_hamming_distances_k)
                    y_mismatch_indexes.append(y_mismatch_indexes_k)

                    list_of_x_predicted.append(copy.deepcopy(x_predicted))

                elif values["-DECODING MODE-"] == "run for each restart":

                    for k in range(r):
                        x_predicted, x_hamming_distances_k, x_mismatch_indexes_k, y_hamming_distances_k, y_mismatch_indexes_k = HMM.viterbi_algorithm(list_of_x_real[k],list_of_y[k],N,T,I,estimated_parameters[k],estimator,values)

                        x_hamming_distances.append(x_hamming_distances_k)
                        x_mismatch_indexes.append(x_mismatch_indexes_k)

                        # print(f"{k} {[round(x_hamming_distances_k[i],3) for i in range(len(x_hamming_distances_k))]}, {round(np.mean(x_hamming_distances_k),2)}, {round(np.var(x_hamming_distances_k),2)}")

                        y_hamming_distances.append(y_hamming_distances_k)
                        y_mismatch_indexes.append(y_mismatch_indexes_k)

                        list_of_x_predicted.append(copy.deepcopy(x_predicted))

                # --------------- decoding algorithm visualisation ---------------

                cs_groups = HMM.define_groups_of_crossed_states(I,N)

                if len(sum([cs_groups[k] for k in range(2,len(cs_groups))],[])) != 0:
                    # replace empty groups by ""
                    cs_dataframe_groups = [cs_groups[i] if len(cs_groups[i]) != 0 else "" for i in range(len(cs_groups))]

                    dataframe = pd.DataFrame()
                    for i in range(len(cs_groups)):
                        dataframe[f"  $G_{i}$   "] = [cs_dataframe_groups[i]]

                    HMM.display_table(
                        dataframe.transpose(), 
                        title=None,
                        figsize=(2,3),
                        colWidths=None, 
                        rowLabels=dataframe.transpose().index, 
                        colLabels=["Індекси"],
                        colColours=None,
                        cellColours=None,
                        savename="table",
                        bbox=[0.25, 0, 0.9, 1]
                    )

                    HMM.display_x_viterbi_results_by_groups_of_mismatch_indices(x_hamming_distances,x_mismatch_indexes,cs_groups,N,T,r)

                if len(sum([cs_groups[k] for k in range(2,len(cs_groups))],[])) == 0:
                    HMM.display_x_viterbi_results_by_mismatch_indices(x_hamming_distances,x_mismatch_indexes,cs_groups,N,T,r)

                HMM.display_y_viterbi_results_by_mismatch_indices(y_hamming_distances,y_mismatch_indexes,I,T,r)

                x_viterbi_results = cv2.imread("Images/x_viterbi_results.png")
                y_viterbi_results = cv2.imread("Images/y_viterbi_results.png")

                if x_viterbi_results.shape != y_viterbi_results.shape:
                    scale = (5.7/7)*x_viterbi_results.shape[1]/y_viterbi_results.shape[1]
                    
                    y_viterbi_results = cv2.resize(
                        y_viterbi_results,
                        (int(y_viterbi_results.shape[1]*scale), int(y_viterbi_results.shape[0]*scale)),
                        interpolation = cv2.INTER_AREA,
                    )

                    y_viterbi_results = cv2.copyMakeBorder(
                        src = y_viterbi_results, 
                        top = 0, 
                        bottom = 0, 
                        left = 15, 
                        right = x_viterbi_results.shape[1] - y_viterbi_results.shape[1] - 15, 
                        borderType = cv2.BORDER_CONSTANT, 
                        value = (255,255,255),
                    )
                    
                    plt.figure(figsize=(15,10.08))
                else:
                    plt.figure(figsize=(12.8,10))

                viterbi_results = np.concatenate((x_viterbi_results, y_viterbi_results), axis=0)

                cv2.imwrite("Images/viterbi_results.png", viterbi_results)

                plt.axis("off")
                plt.tight_layout()
                plt.imshow(viterbi_results)
                plt.show()
                plt.close()

                end_time = time.time()

                decoding_algorithm_indicator = True # decoding algorithm is done
            else:
                sg.Popup("\nRun learning algorithm!\n", title="Error window", custom_text="Close", image="Images/oops.png", button_color="red", font=font)
        except Exception as error:
            sg.Popup(
                f"\n{error}\n",
                title="Error window", 
                image="Images/error.png",
                custom_text="Close",
                button_color="red", 
                font=("Arial", 16)
            )

    elif event == "-ESTIMATE IMPLICIT INDEXES-":
        try:        
            if decoding_algorithm_indicator == True and values["-IMPLICIT INDEXES-"] != "none": 
                window["-ESTIMATED IMPLICIT INDEXES-"].update("I* =")

                real_implicit_indexes = eval(values["-IMPLICIT INDEXES-"])

                implicit_indexes_figure, tabulated_dataframe, estimated_implicit_indexes = HMM.display_predicted_implicit_indices(list_of_x_real,list_of_x_predicted,estimated_parameters,real_implicit_indexes,0,T,N)  

                # window["-ESTIMATED IMPLICIT INDEXES-"].update(r"I* = (1,2,5)")
                window["-ESTIMATED IMPLICIT INDEXES-"].update(f"I* = {estimated_implicit_indexes}")

                plt.figure(figsize=(15,5))
                plt.axis("off")
                plt.tight_layout()
                plt.imshow(implicit_indexes_figure)
                plt.show()
                plt.close()              
                
            else:    
                sg.Popup("\nInput I* value(s) and run decoding algorithm!\n", title="Error window", custom_text="Close", image="Images/oops.png", button_color="red", font=font)
        except Exception as error:
            sg.Popup(
                f"\n{error}\n",
                title="Error window", 
                image="Images/error.png",
                custom_text="Close",
                button_color="red", 
                font=("Arial", 16)
            )

window.close()