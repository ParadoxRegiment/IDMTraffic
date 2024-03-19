import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.integrate import odeint, solve_ivp
import pathlib

class Traffic:
    """ Traffic simulation class based off of the Intelligent Driver Model (IDM).
        Detailed information can be found here:
        https://arxiv.org/abs/cond-mat/0002177
        https://link.springer.com/book/10.1007/978-3-642-32460-4
        https://en.wikipedia.org/wiki/Intelligent_driver_model
        
        A fully animated simulation with multiple lanes can be found here:
        https://www.mtreiber.de/trafficSimulationDe_html5/index.html
    """
    
    def __init__(cls):
        cls.vel_0 = 28 # Init velocity (m/s)
        cls.safe_time = 1.8 # Safe time (s)
        cls.min_gap = 2 # Minimum bumper-to-bumper gap (m)
        cls.accelconst = 0.3 # Acceleration constant (m/s^2)
        cls.decel = 3 # Deceleration constant (m/s^2)
        cls.veh_len = 5 # Vehicle length (m)
        cls.delta = 4 # Accerlation delta (None)
        cls.circ = 1000 # Road circumference (m)
        cls.radius = cls.circ/(2*np.pi)
        cls._randmod = np.random.randint(low=1, high=10) # Randomizer modifier

    def _dyndistance(cls, velnext : float, velcurr : float) -> float:
        """ Calculates the dynamic distance variable.

        Args:
            velnext (float): The velocity of the next car
            velcurr (float): The velocity of the current car

        Returns:
            float: The calculated dynamic distance
        """
        dyndist = cls.min_gap + velcurr*cls.safe_time + \
                    ((velcurr - velnext)/(2*np.sqrt(cls.accelconst*cls.decel)))
        return dyndist

    def _bumpergap(cls, posnext : float, poscurr : float, count : int) -> float:
        """ Calculates the bumper-to-bumper gap between each car.

        Args:
            posnext (float): The position of the next car
            poscurr (float): The position of the current car

        Returns:
            float: The calculated bumper-to-bumper gap between cars
        """
        if count == 0:
            gaparr = cls.circ + posnext - poscurr - cls.veh_len
        else:
            gaparr = posnext - poscurr - cls.veh_len
        
        # Checks if the calculated bumper gap is 0
        # if so adds the randomizer modifier to it
        if np.round(gaparr,3) == 0:
            gaparr += cls._randmod
        return gaparr

    def singlecar(cls, pos_init : float, vel_init : float, time_limit = 900):
        """ IDM traffic model for a single car. Outputs a line graph
            of the position and velocity of the car vs time.

        Args:
            pos_init (float): The initial position of the car
            vel_init (float): The initial velocity of the car
            time_limit (int, optional): The upper time limit for the
                                        simulation in s. Defaults to 900.
        """
        def singlecalcs(datarr, time):
            # datarr will be a 1x2 array, with the second [1] cell
            # being the velocity cell
            vel = datarr[1]
            
            # Derivative of x is velocity, thus sets dx to velocity
            dx = vel
            
            # Calculates the velocity derivative
            dv = cls.accelconst*(1 - (vel/cls.vel_0)**cls.delta)
            
            # Creates an array of the derivates
            rate = np.array([dx, dv])
            
            return rate

        # Creates an initial conditions array made of the pos_init
        # and vel_init arguments
        init_cond = np.array([pos_init, vel_init])

        Tstart = 0
        Tend = time_limit

        # Creates a linespace array from Tstart (0) to Tend
        T = np.linspace(Tstart, Tend, Tend)

        # Calls and runs scipy.odeint on singlecalcs() with init_cond
        # for initial conditions and T for the time array
        X = odeint(singlecalcs, init_cond, T)

        y = X[:,0]
        v = X[:,1]

        plt.plot(T, y, label = "Position")
        plt.plot(T, v, label = "Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity or Position (m or m/s)")
        plt.legend()
        plt.show()
        
    def multicar(cls, car_count : int, graph_type : int, output : str, frames_count = 90,
                 time_limit = 900, pos_type = 'e', vel_type = '0'):
        """ IDM traffic model for multiple cars. Outputs a graph--either
            a straight line or circle, depending on the value passed into
            graph_type.

        Args:
            car_count (int): The number of cars to be used in the simulation
            graph_type (int): Determines the type of graph to be shown
            time_limit (int, optional): The upper time limit for the
                                        simulation in s. Defaults to 900.
        """
        # Sets the initial position of every car to either a linear range of numbers
        # or an array of random numbers with the min_gap distance and vehicle length
        # added as well.
        match pos_type.lower():
            case 'e' | 'even': 
                pos_init = np.arange(car_count, 0, -1) * (cls.min_gap + cls.veh_len)
            case 'r' | 'random' | 'rand':
                pos_init = np.empty(car_count)
                for i in range(car_count):
                    pos_init[i] = np.round((np.random.randint(low=1, high=cls.circ) + cls.min_gap + cls.veh_len), decimals=3)
                    if pos_init[i] == pos_init[i-1]:
                        pos_init[i] += np.random.randint(low=1, high=10)
        
        # Sets the initial velocity of every car to either an array of zeros
        # or to an array of random numbers
        match vel_type.lower():
            case '0' | 'z' | 'zeros':
                vel_init = np.zeros(car_count)
            case 'r' | 'random' | 'rand':
                vel_init = np.random.randint(low=10, high=30, size=car_count)
        
        # Dictionary to be used in the generation of the animation gifs
        cls._graph_dict = dict.fromkeys(['r', 'rand', 'random'], 'Random')
        cls._graph_dict.update(dict.fromkeys(['0','z','zeros'], 'Zeros'))
        cls._graph_dict.update(dict.fromkeys(['e','even'], 'Even'))
        
        def multicalcs(datarr, time):
            # Sets the position and velocity arrays according to the number
            # of cars passed into car_count
            pos = datarr[:car_count]
            vel = datarr[car_count:]
            
            dv = np.zeros(car_count)
            
            for i in range(car_count):
                if i == 0:
                    # Calculates the decel parameter for the first car
                    decelparam = cls._dyndistance(velcurr=vel[i], velnext=vel[i-1])\
                                /cls._bumpergap(poscurr=pos[i], posnext=pos[-1], count = i)
                else:
                    # Calculates the decel parameters for every other car
                    decelparam = cls._dyndistance(velcurr=vel[i], velnext=vel[i-1])\
                                /cls._bumpergap(poscurr=pos[i], posnext=pos[i-1], count = i)
                    
                    ### Something about this calculation is causing issues with any circumference
                    # other than 1000m. At 100m the first car completely bypasses the last car at
                    # least twice. From at least 200m - at least 900m the first car never even
                    # reaches the last car. And at any circumference above 1000m the same byypassing
                    # behavior can be observed. I'm not sure how to fix this, or if it's just a
                    ### consequence of the IDM setup.
                
                dx = vel
                
                # Calculates the velocity derivative for every car
                dv[i] = cls.accelconst*(1 - (vel[i]/cls.vel_0)**cls.delta - (decelparam)**2)
                
            rate = np.concatenate((dx, dv))
            
            return rate
        
        # Creates an initial conditions 1d array made of the pos_init
        # and vel_init arguments
        init_cond = np.concatenate((pos_init, vel_init))

        Tstart = 0
        Tend = time_limit

        # Creates a linespace array from Tstart (0) to Tend
        time = np.linspace(Tstart, Tend, Tend)
        
        # Calls and runs scipy.odeint() on singlecalcs() with init_cond
        # for initial conditions and T for the time array
        car_pos = odeint(multicalcs, init_cond, time)

        # Sets the rcParams for the animation figures that will be generated
        plt.rcParams['animation.html'] = "jshtml"
        plt.rcParams['figure.dpi'] = 150
        plt.style.use('dark_background')
        plt.ioff()
        
        fig, ax = plt.subplots()
        
        # Animation function for a flat line of points
        def AnimPlot(i):
            xarr = car_pos[i*10][:car_count] % 1000
            yarr = np.ones(car_count)
            
            plt.cla()
            ax.plot(xarr[1:-1], yarr[1:-1], marker = '>', linestyle = '', color = 'w')
            
            # Changes the first and last car points to different colors
            ax.plot(xarr[:1], yarr[:1], marker = '*', color = 'magenta', linestyle = '', markersize = 10)
            ax.plot(xarr[-1:], yarr[-1:], marker = '*', color = 'lime', linestyle = '', markersize = 10)
            plt.xlim(0,1000)
        
        # Animation function for a circle of points
        def CircleAnim(i):
            theta = car_pos[i*10][:car_count]/(1000*2*np.pi)
            xarr = cls.radius*np.cos(theta)
            yarr = cls.radius*np.sin(theta)
            
            ax.set_aspect('equal', adjustable = 'box')
            plt.cla()
            ax.plot(xarr[1:-1], yarr[1:-1], marker = 'D', linestyle = '', color = 'w')
            
            # Changes the first and last car points to different colors
            ax.plot(xarr[:1], yarr[:1], marker = '*', color = 'magenta', linestyle = '', markersize = 10)
            ax.plot(xarr[-1:], yarr[-1:], marker = '*', color = 'lime', linestyle = '', markersize = 10)
            plt.axis([-cls.radius*1.1, cls.radius*1.1, -cls.radius*1.1, cls.radius*1.1])
        
        # Generates and saves a graph as a gif. Graph depends on the
        # value passed into graph_type
        dir_path = pathlib.Path().resolve()
        match output.lower():
            case 'p' | 'plot':
                plt.show()
            case 'g' | 'gif':
                match graph_type:
                    case 0:
                        animation = anim.FuncAnimation(fig, CircleAnim, frames = frames_count)
                        circ_gif_name = (f'Circle{cls.circ}Circu{cls._graph_dict[pos_type.lower()]}'
                                        +f'Pos{cls._graph_dict[vel_type.lower()]}Vel.gif')
                        writer = anim.PillowWriter(fps = 30, bitrate = 1000)
                        animation.save(circ_gif_name, writer=writer)
                        print(circ_gif_name, "saved to", dir_path)
                        plt.show()
                    case 1:
                        animation = anim.FuncAnimation(fig, AnimPlot, frames = frames_count)
                        flat_gif_name = (f'Flat{cls.circ}Circu{cls._graph_dict[pos_type.lower()]}'
                                        +f'Pos{cls._graph_dict[vel_type.lower()]}Vel.gif')
                        writer = anim.PillowWriter(fps = 30, bitrate = 1000)
                        animation.save(flat_gif_name, writer=writer)
                        print(flat_gif_name, "saved to", dir_path)
                        plt.show()                 
            case _:
                print("Unknown output type! Please rerun program!")

test = Traffic()
test.circ = 800
test.multicar(car_count=30, graph_type=0, pos_type='e', vel_type='0', output='gif')
# test.singlecar(pos_init=0, vel_init=30, time_limit=150)
