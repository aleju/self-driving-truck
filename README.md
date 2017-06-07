# About

This repository contains code to train and run a self-driving truck in Euro Truck Simulator 2.
The resulting AI will automatically steer, accelerate and brake.
It is trained (mostly) via reinforcement learning and only has access to the buttons W, A, S and D
(i.e. it can not directly set the steering wheel angle).

Example video:

[![Example video](images/readme/video.jpg?raw=true)](https://www.youtube.com/watch?v=59iNsSnAUfA)

# Architecture and method

The basic training method follows the standard reinforcement learning approach from the original [Atari paper](https://arxiv.org/abs/1312.5602).
Additionally, a separation of Q-values in V (value) and A (advantage) - as described in [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) - is used.
Further, the model tries to predict future states and rewards, similar to the description in [Deep Successor Reinforcement Learning](https://arxiv.org/abs/1606.02396).
(While that paper uses only predictions for the next timestep, here predictions for the next T timesteps are generated via an LSTM.)
To make training faster, a semi-supervised pretraining is applied to the first stage of the whole model (similar to [Loss is its own Reward: Self-Supervision for Reinforcement Learning](https://arxiv.org/abs/1612.07307v2), though here only applied once at the start).
That training uses some manually created annotations (e.g. positions of cars and lanes in example images)
as well as some automatically generated ones (e.g. canny edges, optical flow).

Architecture visualization:

![Architecture](images/readme/architecture.png?raw=true "Architecture")

There are five components:
* Embedder 1: A CNN that is pretrained in semi-supervised fashion. The two gradient inputs (see image) are just gradients from `1` to `0` which are supposed to give positional information. (E.g. the mirrors are always at roughly the same positions, so it is logical to detect them partially by their position.) Instance Normalization was used, because Batch Normalization regularly broke, resulting in zero-only vectors during test/eval (seemed like a bug in the framework, would usually go away when using batch sizes >= 2 or staying in training mode).
* Embedder 2: Takes the results of Embedder 1 and converts them into a vector. Additional inputs are added here. (These are: (1) Previous actions, (2) whether the gear is in reverse mode, (3) steering wheel position, (4) previous and current speeds. The current gear state and the speed is read out from the route advisor. The steering wheel position is approximated using a separate CNN.) Not merging this component with Embedder 1 allows to theoretically keep the weights from pretraining fixed.
* Direct Reward: A model that predicts the direct reward, i.e. for `(s, a, r), (s', a', r')` it predicts `r` when being in `s'`. The reward is bound to the range -100 to +100. It predicts the reward value using a softmax over 100 bins.
* Indirect Reward: A model that predicts future rewards, i.e. for `(s, a, r), (s', a', r'), ...` it predicts `r + gamma*r' + gamma^2*r''` when being in state `s`. Gamma is set to `0.95`. This model uses standard regression. It predicts one value per action, i.e. `Q(s, a)`.
* Successors: An RNN model that predicts future embeddings (when specific actions are chosen). These future embeddings can then be used to predict future direct and indirect rewards (using the two previous models). This module uses an addition to the previously generated embedding (i.e. residual architecture). That way the LSTMs only have to predict the changes (of the embeddings) that were caused by the actions.

Aside from these, there is also an autoencoder component applied to the embeddings of Embedder 2.
However, that component is only trained for some batches, so it is skipped here.

During application, each game state (i.e. frame/screenshot at 10fps) is embedded via convolutions and fully connected layers to a vector.
From that vector, future embeddings (the successors) are predicted.
Each such prediction (for each timestep) is dependent on a chosen action (e.g. pressing W+A followed by two times W converts game state vector X into Y).
For 9 possible actions (W, W+A, W+D, S, S+A, S+D, A, D, none) and 10 timesteps (i.e. looking 1 second into the future), this leads to roughly 3.5 billion possible chains of actions.
This number is decreased to roughly 400 sensible plans (e.g. 10x W, 10x W+A, 3x W+A followed by 7x W, ...).
For each such plan, the successors are generated and rewards are predicted (which can be done reasonably fast as the embedding's vector size is only 512).
The plans are ordered/ranked by the V-values of their last timesteps and the plan with the highest V-value is chosen.
(The predicted direct rewards of successors are currently ignored in the ranking, which seemed to improve the driving a bit.)

# Reward Function

For a chain `(s, a, r, s')`, the reward `r` is mainly dependent on the measured speed at `s'`.
The formula is `r = sp*rev + o + d`, where
* `sp` is the current speed (clipped to 0 to 100),
* `rev` is `0.25` if the truck is in reverse gear ("drive backwards mode") and otherwise `1`
* `o` is `-10` if an offence is shown (e.g. "ran over a red light", "crashed into a vehicle")
* `d` is `-50` if the truck has taken damage (shown by a message similar to offences, which stays for around 2 seconds)

The speed is read out from the game screen (it is shown in the route advisor).
Similarly, offences and damages can be recognized using simple pixel comparisons or instance matching in the area of the route advisor (both events lead to shown messages).

# Difficulties

ETS2 is a harder game to play (for an AI) than it may seem at first glance. Some of the difficulties are:
* The truck's acceleration is underwhelming. The AI has to keep buttons pressed for a long time to slowly accumulate speed, especially at hills. This makes it also quite bad whenever the AI crashes into something, as getting out of that situation will require to again accumulate speed.
* Same is the case for braking.
* The truck has separate forward and reverse gears. You can only drive backwards when you are in reverse mode (or forward in the forward gear). To switch gears, the speed has to be flat zero, with no buttons pressed for a short amount of time, then forward/backward will activate forward/backward mode (in the simplest settings). Especially the flat zero speed is hard to achieve for the AI (see also slow acceleration/braking).
* The truck needs a very large area to turn around.
* When driving backwards, the front of the truck can collide with the container at the back. Then it can't drive further backwards, limiting the maximum length that you can drive that way before having to switch again to forward+left/right (depending on the situation).
* When pressing left or right, the truck does not immediately drive towards the left or right. Instead, these keys merely change the position of the steering wheel. It is common to steer (via left/right button) into a direction for several timesteps (e.g. one second), while the truck keeps driving into the opposite direction the whole time.
* The steering wheel can turn by more than 360 degrees, so when it looks like you are steering towards the right you might actually be steering towards the left. (Here, the steering wheel is tracked at all times via a CNN that runs separately from the AI's CNN - though that model sometimes produces errors.)
* The truck can get randomly stuck on objects that aren't even visible from the driver's seat. This may include objects blocking some wheel at the far back of the container. Even as a human driver, that can be annoying and hard to fix.
* Driving into the green can get you stuck at invisible walls (or tiny hills, see above point), which is hard to understand for the AI. (Especially as it has almost no memory - so it might not even recognize in any way that it is stuck.)
* Sidenote: Most of the game is spent pressing only W (forward) on the highway, which makes it hard to train a model in supervised fashion from recorded gameplay (`W+no steering` has basically always maximum probability).

And of course on top of these things, the standard problems with window handling occur (e.g. where is the window; what pixel content does it show; is the game paused; at what speed is the truck driving; how to send keypresses to the window etc.). Also, the whole architecture has to run in (almost-)realtime (theoretically here max 100ms per state, but better <=50ms so that actions have a decent effect before observing the next screen).

# Limitations

The AI can -- to a degree -- drive on streets that have solid, continuous objects on both sides.
On such roads, hitting the side is usually no death sentence as the truck is deflected from the wall and can go on driving (albeit damaged).
That is in contrast to e.g. streets without any objects on the side, where the AI can drive off the street and then get stuck on some tiny hill/object or run into an invisible wall.
As a consequence, the AI is best at driving on highways, which usually have such walls or railings on both sides (and are often quite wide).
However, it will not care about lanes and not that much about other cars (but it seems to recognize them).
In general, the AI's driving capabilities are still far away from the ones of humans.

Typical problems of the AI are:
* Steering towards a wall: Sometimes it will hit a wall at a diagonal angle and continue to steer into it. E.g. the wall is on the right side and the AI choses to steer right instead of left. The truck will then come to a standstill and not move any more. The reason for why the AI does that is unclear -- might be connected to the steering wheel being able to turn by more than 360 degrees, creating confusion during the training. Or maybe it correlates some random pattern on the screen with driving towards the right, instead of just looking at the street and walls. Another possible cause is that driving towards the center of the street in these situations has a high likelihood of causing crashes with other cars, as these approach from behind with high speed. These crashes induce significant negative reward, which might lead the AI to prefer the permanent zero-reward from the wall.
* Driving frontally into a wall: It can happen that the truck drives frontally into a wall, though not so often (more common with lanterns or railings in cities). Sometimes it will switch into reverse mode to get out of the situation.
* Small objects: It easily gets stuck on these and then can't see them from the driver's seat. As its memory is only 200ms, it will usually not switch into reverse gear. This also happens on highways at the start of railings.
* Lamps and lanterns: Same problem as with small objects.
* Driving into the green: When there is no wall/railing on the side of the road, it is just a matter of time until it will drive off-road. From there it will often not recover.
* Intersections: It will usually try to just drive forward. If there's a wall/railing there (T-intersection), it will often just drive frontally into it.
* Fuel stations and rest areas: The AI was trained on a highway that didn't have these. Now it seems to be quite confused by them. When it sees these, there is probably a >50% chance that it will drive towards the the fuel station or rest zone and usually hit an object or the green.
* Other cars: It seems to kind of recognize these on highways. Sometimes it will overly try to evade them, leading to crashes. In city areas (where other cars don't drive in the same direction), it will sometimes completely ignore them, resulting in frontal crashes.

Note that all of this is based on the results after a few days of training.
More training might yield better behavior.

# Usage

## Requirements and dependencies

Hardware
* Nvidia GPU with 8GB+ memory
* 32GB+ of RAM (might need less to only apply the model)

System
* Ubuntu with GUI (this requires an X-Server, i.e. it will not run in Windows)
* python 2.7 (only tested in this version; some stuff was adapted to be py3 ready)
* xwininfo (should be installed by default in Ubuntu)
* xdotool (`sudo apt-get install xdotool`)
* python-xlib (`sudo apt-get install python-xlib`)
* CUDA with CUDNN

Python libraries
* imgaug
* skimage
* pytorch (must be commit `2acfb23` or newer)
* scipy
* numpy
* PIL
* cv2 (OpenCV)
* sqlite3 (usually installed by default, might need some system packaged though)
* matplotlib
* Tkinter
* PyUserInput
  * `git clone https://github.com/PyUserInput/PyUserInput.git`
  * `python setup.py sdist`
  * `sudo pip install dist/PyUserInput-0.1.12.tar.gz`

## Install

* `cd ~/`
* `git clone https://github.com/aleju/self-driving-truck.git`
* `cd self-driving-truck && cd lib && gcc -shared -O3 -Wall -fPIC -lX11 -Wl,-soname,screenshot -o screenshot.so screenshot.c -lX11`
* Download the models:
  * [https://drive.google.com/open?id=0B2MhAqRpz7P9MXYycU1ONFVaWHM](train_reinforced_model.tar.gz)
  * [https://drive.google.com/open?id=0B2MhAqRpz7P9VWNVdmhabzlRc28](train_semisupervised_model.tar.gz) (optional, only needed if restarting the reinforcement learning)
  * [https://drive.google.com/open?id=0B2MhAqRpz7P9TEtZOFAwMkVXY1k](train_semisupervised_model_withshortcuts.tar.gz) (very optional, was only only used to create the video)
* Decompress the downloaded models to `.tar` files.
* Copy `train_reinforced_model.tar` to `train_reinforced/train_reinforced_model.tar`.
* Copy `train_semisupervised_model.tar` to `train_semisupervised/train_semisupervised_model.tar`.
* Copy `train_semisupervised_model_withshortcuts.tar` to `train_semisupervised/train_semisupervised_model_withshortcuts.tar`.
* Install Steam for Linux
* Install ETS2 (Demo is enough)

## Game Configuration

* Start a new profile (default truck) and chose the easiest gear for the truck.
* Switch to the driver seat's camera (press `1`), activate the right side mirror (press `F2`) and the route advisor (press `F3`).
* Make N save games at varying locations. Open `config.py` and change `RELOAD_MAX_SAVEGAME_NUMBER` to `N`.

* The directory `other/config_files` of the repository contains the configuration used during the training (main config file of the game, profile config file, profile control settings). These files can optionally be copied to the game. Otherwise the used settings are below.

* Keyboard settings:
  * Put pause on `F1` key and quick load on `F11`.
  * Forward must be on `W`, backward on `S`, steering left on `A` and steering right on `D`.

* Graphics settings (some checks work pixel accurate, so your settings MUST be identical to these, especially the resolution):
  * Deactivate full screen mode
  * Brightness: middle (default?)
  * Resolution: 1280x720
  * Refresh rate: Default
  * Vertical synchronization: Enabled
  * Scaling: 50%
  * Set all checkboxes below `Scaling` to `off`, except `Normal Maps` (`on`)
  * Set all select boxes below `Scaling` to `Low` or `Disabled`
  * Anisotropic filtering: Lowest value (far left)

* Relevant gameplay settings (mostly default):
  * Game settings
    * Fatigue simulation: off
    * Traffic offense: on
    * Route advisor speed limit: Show truck limit
    * Route advisor speeding warning: on
    * Keep route advisor hidden: off
    * Show navigation: always
  * Truck settings
    * Transmission type: Simple automatic (important!)
    * Steering autocenter: on
    * Braking intensity: middle
    * Trailer stability: middle
    * Truck speed limiter: off
    * Automatic engine brake: off
    * Automatic engine and electricity start: on
  * Camera settings
    * Steering camera rotation: off
    * Steering camera rotation factor: middle
    * Steering camera rotation on reverse: Normal
    * Blinker camera rotation: off
    * Physical camera movement: on
    * Physical camera movement factor: middle
  * Regional settings
    * Language: English
    * Length units: kilometers (important!)

* Relevant Controls settings:
  * Adaptive automatic transmission: off
  * Steering sensitivity: about 1/4
  * Steering non-linearity: none (far left)

* Screenshots with game settings:
  * Graphics: [1](images/readme/settings_graphics_1.jpg) [2](images/readme/settings_graphics_2.jpg)
  * Gameplay: [1](images/readme/settings_gameplay_1.jpg) [2](images/readme/settings_gameplay_2.jpg) [3](images/readme/settings_gameplay_3.jpg) [4](images/readme/settings_gameplay_4.jpg)
  * Keys & Buttons: [1](images/readme/settings_keys_and_buttons_1.jpg) [2](images/readme/settings_keys_and_buttons_2.jpg) [3](images/readme/settings_keys_and_buttons_3.jpg) [4](images/readme/settings_keys_and_buttons_4.jpg) [5](images/readme/settings_keys_and_buttons_5.jpg) [6](images/readme/settings_keys_and_buttons_6.jpg)
  * Controls: [1](images/readme/settings_controls_1.jpg) [2](images/readme/settings_controls_2.jpg) [3](images/readme/settings_controls_3.jpg) [4](images/readme/settings_controls_4.jpg)

## Apply model

* Start ETS2.
* Move the game window to the bottom right corner (to have enough space for the terminal and overview window at top and left).
* The game window must be fully on screen.
* Start a game so that the truck can be driven by the AI.
  * The training happened mainly on the highway near Kassel, in direction of Frankfurt/Erfurt. The AI will play best there.
  * Other highways may or may not cause problems. Driving performance in cities will generally be poor.
* Make sure that your game looks similar to the one in the video.
  * Resolution must be `1280x720`.
  * Camera perspective must be from the driver's seat.
  * Mirror at the top right must be visible.
  * Steering wheel must be visible.
  * Route advisor must be visible. It must contain your speed. Speed must be in `km/h`.
  * The truck must be controlable via WASD.
  * Turning the steering wheel must feel rather slow (default setting). No fast turning.
  * The ingame time should not be at dawn, dusk or night. These can cause problems.
* Move the steering wheel to 0 degrees (straight driving).
* `cd train_reinforced`
* `python train.py --onlyapply --noinsert --p_explore=0.1`
* Do not move the game window any more.
* Keep the game window activated, the AI should now drive automatically.

## Run full training

* Notes:
  * For full training, knowledge of CNNs and RL is strongly recommended as you might have to change learning rates, gradient weightings or other stuff.
  * These steps are not thoroughly tested, they may contain bugs or throw errors.
  * Training will likely take several days.

* Steps:
  * Download the annotations from [here](https://drive.google.com/open?id=0B2MhAqRpz7P9TkJxQUhfQzg1NUE), decompress them to `.pickle` and put them into `annotations/`.
  * Run the game (as described above).
  * `cd scripts`
  * `python collect_experiences_supervised.py`
    * Wait for the script to fully start, then switch to the game and activate tracking by pressing `CTRL` (the window should show that it is saving new memories).
    * Truck around until you have at least about 25000 memories (at 10 per second that is about 0.7 hours).
    * You may check in between that the file `replay_memory_supervised.sqlite` is really growing in size.
    * Once you have enough memories added, switch back to the terminal and press `CTRL+C` to end the script.
  * `python collect_experiences_supervised.py --val`
    * Repeat the above steps to add some validation memories. A few thousand should be enough.
  * `cd ../train_semisupervised`
  * `python train.py --nocontinue`
    * Run for something around 5000 to 30000 batches. Then CTRL+C to end training.
  * `cd ../train_reinforced`
  * `python train.py --nocontinue`
    * As above, wait for the script to start, then switch to the game. The AI should be steering the truck randomly.
    * Let this train for several days.
    * The directory should contain some graphics (loss plots and debug images) that are generated during training. Check them to verify that everything is going well. Loss values can increase during training when new experiences were gathered.
