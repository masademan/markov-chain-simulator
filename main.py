import os
import sys
import json
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# State object
#  attributes:
#   dictionary with next state objects and respective probabilities
#   state name
#   shorter state name
#   special meaning (losing, winning, etc.)
#  methods:
#   object.sample_next_state()
#   object.add_next_state(state, probability) (adds a new next state to the dictionary)
#   object.normalize() (normalizes the probabilities)
#   objectA == objectB
#   hash(object)

class State:
    def __init__(self, state_name, shorthand_state_name, special_meaning=""):
        self.state_name_to_probability = {}
        self.state_name_to_state = {}
        self.state_name = state_name
        self.shorthand_state_name = shorthand_state_name
        self.special_meaning = special_meaning

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return self.state_name_to_probability.items() == other.state_name_to_probability.items() and \
        self.state_name == other.state_name and self.shorthand_state_name == other.shorthand_state_name and \
        self.special_meaning == other.special_meaning
    
    def __str__(self):
        str_dict = ["{}"]
        TAB = "  "

        if len(self.state_name_to_probability) > 0:
            str_dict = ["{\n"]
            for key, value in self.state_name_to_probability.items():
                str_dict.append(f"{TAB}{key}: {value}\n")
            str_dict.append("}")

        return f"{self.state_name} ({self.shorthand_state_name}): {"".join(str_dict)}"
    
    @classmethod
    def name_from_tuple(cls, input_tuple):
        if len(input_tuple) < 2:
            return NotImplemented
        
        if len(input_tuple) == 2:
            return cls(input_tuple[0], input_tuple[1])
        return cls(input_tuple[0], input_tuple[1], input_tuple[2])

    def normalize(self, override_precision=None):
        if self.state_name_to_probability == {}:
            return False

        probability_factor = sum(self.state_name_to_probability.values())

        if probability_factor == 1:
            return True

        if abs(1 - probability_factor) > 0.001:
            precision = 2 if override_precision is None else override_precision
            for key in self.state_name_to_probability:
                current_num = self.state_name_to_probability[key] / probability_factor * 10**precision
                if current_num >= 0.5:
                    current_num = int(current_num) + 1
                else:
                    current_num = int(current_num)
                self.state_name_to_probability[key] /= current_num / 10**precision

        if sum(self.state_name_to_probability.values()) == 1:
            return True

        max_key = None
        max_prob = 0

        for key, value in self.state_name_to_probability.items():
            if value > max_prob:
                max_key = key
                max_prob = value
        
        self.state_name_to_probability[max_key] = 1 - probability_factor + self.state_name_to_probability[max_key]
        return True

    def add_state_connection(self, state, probability):
        if state.state_name in self.state_name_to_probability:
            return False
        
        self.state_name_to_probability[state.state_name] = probability
        self.state_name_to_state[state.state_name] = state
        return True

    def sample_next_state(self):
        if not self.normalize():
            return self

        possible_states = list(self.state_name_to_probability.keys())
        probabilities = list(self.state_name_to_probability.values())

        return self.state_name_to_state[random.choices(possible_states, weights=probabilities, k=1)[0]]
    
# Trial object
#  attributes:
#   max runs
#   num runs done
#   starting state
#   special meanings to stop at (list of strings, i.e., ["winning", "losing"]) (overrides the 0 verbosity)
#   special meanings to print at (list of string, i.e., ["almost losing", "almost winning"]) (overrides the 0 verbosity)
#   verbosity (0, 1, or 2)
#  methods:
#   object.step()
#   object.run_trial()
#   object.copy()
#   Trial.copy(object) (@classmethod)

class Trial:
    def __init__(self, start_state, special_stops=[], special_prints=[], max_runs=1_000, verbosity=1, stop_warning=False):
        self.start_state = start_state
        self.current_state = start_state
        self.special_stops = special_stops
        self.special_prints = special_prints
        self.max_runs = max_runs
        self.verbosity = verbosity

        self.num_steps_done = 0
        self.currently_running = True

        self.special_print_visits = {special_print: 0 for special_print in special_prints}

        if len(special_stops) == 0 and not stop_warning:
            input("WARNING: The length of special_stops is 0, which means this Trial will never end, press enter to continue")

    @classmethod
    def copy(cls, other):
        if not isinstance(other, Trial):
            return NotImplemented
        return other.copy()
    
    def copy(self):
        return Trial(self.start_state, self.special_stops.copy(), self.special_prints.copy(), self.max_runs, self.verbosity, True)
    
    def step(self, override_verbosity=None):
        verbosity = self.verbosity
        if override_verbosity is not None:
            verbosity = override_verbosity

        prev_state = self.current_state
        self.current_state = self.current_state.sample_next_state()
        if prev_state != self.current_state:
            self.num_steps_done += 1
        suffix = "" if verbosity == 0 else f" after {self.num_steps_done} step(s)"

        if self.current_state.special_meaning in self.special_stops:
            self.currently_running = False
        elif self.current_state.special_meaning in self.special_prints and verbosity != 0:
            print(f"Reached '{self.current_state.state_name}' state with special meaning '{self.current_state.special_meaning}'{suffix}")
            self.special_print_visits[self.current_state.special_meaning] += 1
        elif verbosity == 2:
            print(f"Moved to '{self.current_state.state_name}' state{suffix}")

        if not self.currently_running and verbosity != 0:
            print(f"Ended trial at '{self.current_state.state_name}' with special meaning '{self.current_state.special_meaning}'{suffix}")

    def run_trial(self, override_verbosity=None):
        self.currently_running = True

        while self.currently_running and self.num_steps_done < self.max_runs:
            self.step(override_verbosity)

# Simulation object
#  attributes:
#   trial to run multiple times
#   data (starts as none) (a dict of num runs done and how many time it's been done)
#  methods:
#   object.run_simulation(num times, verbosity, add to data) (adds on to the existing data if add to data is true, otherwise, it'll output the new data)
#   object.create_graph()
#   object.add_new_data(new data)
#   object.get_mean()
#   object.get_standard_deviation()

class Simulation:
    def __init__(self, trial_to_run, data={}):
        self.starting_trial = trial_to_run
        self.trial_to_run = trial_to_run.copy()

        self.data = data

    @classmethod
    def from_state(cls, state, special_stops=[], special_prints=[], max_runs=1_000, verbosity=1):
        if len(special_stops) == 0:
            input("WARNING: The length of special_stops is 0, which means this Trial will never end, press enter to continue")
            
        return cls(Trial(state, special_stops, special_prints, max_runs, verbosity, True))
    
    @classmethod
    def from_file(cls, file_path):
        if not os.path.exists(file_path):
            print("THIS FILE PATH DOES NOT EXIST", file=sys.stderr)
            sys.exit(-1)

        with open(file_path, "r") as jsonFile:
            data_dict = json.load(jsonFile)
        
        starting_state = create_connected_states(data_dict["trial_dict"]["connections"], data_dict["trial_dict"]["starting_state_name"], True)

        if len(data_dict["trial_dict"]["special_stops"]) == 0:
            input("WARNING: The length of special_stops is 0, which means this Trial will never end, press enter to continue")
            
        return cls(Trial(
            starting_state,
            data_dict["trial_dict"]["special_stops"],
            data_dict["trial_dict"]["special_prints"],
            data_dict["trial_dict"]["max_runs"],
            data_dict["trial_dict"]["verbosity"],
            True,
        ),
            {int(num): data_dict["data"][num] for num in data_dict["data"]},
        )
    
    def to_file(self, file_path, human_readable=False):
        file_dict = {
            "trial_dict": {
                "connections": extract_connection_dict(self.trial_to_run.start_state, True),
                "starting_state_name": self.trial_to_run.start_state.state_name,
                "special_stops": self.starting_trial.special_stops,
                "special_prints": self.starting_trial.special_prints,
                "max_runs": self.starting_trial.max_runs,
                "verbosity": self.starting_trial.verbosity,
            },
            "data": self.data,
        }

        with open(file_path if file_path.endswith(".json") else file_path + ".json", "w") as jsonFile:
            if human_readable:
                jsonFile.write(json.dumps(file_dict, indent=4, sort_keys=True))
            else:
                json.dump(file_dict, jsonFile)

    def run_simulation(self, num_trials, override_verbosity=None, add_to_data=True):
        data_found = {}

        for _ in range(num_trials):
            self.trial_to_run = self.starting_trial.copy()
            self.trial_to_run.run_trial(override_verbosity)

            num_steps_in_trial = self.trial_to_run.num_steps_done
            if num_steps_in_trial not in data_found:
                data_found[num_steps_in_trial] = {special_state: 0 for special_state in (self.starting_trial.special_stops + self.starting_trial.special_prints)}
            data_found[num_steps_in_trial][self.trial_to_run.current_state.special_meaning] += num_steps_in_trial
            for special_print in self.starting_trial.special_prints:
                data_found[num_steps_in_trial][special_print] += self.trial_to_run.special_print_visits[special_print]
        
        if add_to_data:
            self.add_new_data(data_found)
            return
        
        return data_found

    def add_new_data(self, data_to_add):
        old_keys = self.data.keys()
        new_keys = data_to_add.keys()

        if len(new_keys) == 0:
            return
        
        min_num_moves = min(new_keys)
        max_num_moves = max(new_keys)
        if len(old_keys) > 0:
            min_num_moves = min(min(old_keys), min_num_moves)
            max_num_moves = max(max(old_keys), max_num_moves)

        for num_moves in range(min_num_moves, max_num_moves + 1):
            all_special_states = self.starting_trial.special_stops + self.starting_trial.special_prints
            if num_moves not in self.data:
                self.data[num_moves] = {special_state: 0 for special_state in all_special_states}
            for special_state in all_special_states:
                if num_moves in data_to_add:
                    self.data[num_moves][special_state] += data_to_add[num_moves].get(special_state, 0)

    def create_graph(self, full_plot = False):
        if self.data == {}:
            print("No data inputted yet")
            return
        
        num_moves = list(map(str, sorted(self.data.keys())))

        for special_stop in self.starting_trial.special_stops:
            used_special_prints = []
            name_and_frequencies = []

            for num_move in num_moves:
                frequency = self.data[int(num_move)].get(special_stop, 0)
                if frequency != 0 or full_plot:
                    used_special_prints.append(num_move)
                    name_and_frequencies.append(frequency)

            plt.bar(used_special_prints, name_and_frequencies)

            plt.xticks(rotation=45)
            plt.xlabel("Num moves")
            plt.ylabel("Frequency")
            plt.title(f"Num moves vs Frequency of the ending '{special_stop}'")
            maximize_plt_plot()
            plt.show()

        name_and_frequencies = {}
        for special_print in self.starting_trial.special_prints:
            for num_move in num_moves:
                frequency = self.data[int(num_move)].get(special_print, 0)
                if frequency != 0 or full_plot:
                    if special_print not in name_and_frequencies:
                        name_and_frequencies[special_print] = frequency

        plt.bar(name_and_frequencies.keys(), name_and_frequencies.values())

        plt.xticks(rotation=45)
        plt.xlabel("Special prints")
        plt.ylabel("Num visits")
        plt.title(f"Num visits vs Special print states visited")
        maximize_plt_plot()
        plt.show()

    def get_data_with_keys(self, keys_to_use=None):
        keys_to_use = self.starting_trial.special_stops + self.starting_trial.special_prints if keys_to_use is None else keys_to_use
        keyed_dataset = []

        for num_moves, frequencies in self.data.items():
            for label in keys_to_use:
                keyed_dataset += [num_moves] * frequencies[label]

        return keyed_dataset

    def get_mean(self, keys_to_use=None):
        keyed_data = self.get_data_with_keys(keys_to_use)

        return sum(keyed_data) / len(keyed_data)

    def get_standard_deviation(self, keys_to_use=None):
        keyed_data = np.array(self.get_data_with_keys(keys_to_use))

        return (sum((keyed_data - self.get_mean(keys_to_use)) ** 2) / len(keyed_data)) ** 0.5

# function
#  inputs:
#   a dictionary of form dict[(name, shorter name, meaning), [(next state name a, probability a), (next state name b, probability b), etc...]]; dict[tuple[str, str, str|None], list[tuple[str, float]]]
#   state to start on
#  output:
#   a State object that's the state that you want to start on, with all the correct connections built

def JSON_connections_to_dict(json_connections):
    converted_dict = {}

    for state_names in json_connections:
        current_state_key = tuple(state_names.split("|"))
        converted_dict[current_state_key] = []
        for connection in json_connections[state_names]:
            connection_data = connection.split("|")
            converted_dict[current_state_key].append((connection_data[0], float(connection_data[1])))
    
    return converted_dict

def create_connected_states(connections, starting_state, no_tuples=False):
    if no_tuples:
        connections = JSON_connections_to_dict(connections)

    # Create states
    state_name_to_state = {}

    for state in connections:
        state_name_to_state[state[0]] = State.name_from_tuple(state)

    # Connect states
    for source, edges in connections.items():
        for name, probability in edges:
            if name not in state_name_to_state:
                print(f"Name '{name}' isn't a defined state", file=sys.stderr)
                sys.exit(-1)
            if not state_name_to_state[source[0]].add_state_connection(state_name_to_state[name], probability):
                print(f"WARNING: State '{name}' was already connected to state '{source[0]}', so it was skipped")
        state_name_to_state[source[0]].normalize()

    if starting_state not in state_name_to_state:
        print(f"The starting_state '{starting_state.state_name}' is not a defined state", file=sys.stderr)
        sys.exit(-1)

    return state_name_to_state[starting_state]

# function
#  inputs:
#   State object (this is where you extract the connection dict)
#  output:
#   a dictionary of form dict[(name, shorter name, meaning), [(next state name a, probability a), (next state name b, probability b), etc...]]; dict[tuple[str, str, str|None], list[tuple[str, float]]]

def extract_connection_dict(starting_state, no_tuples=False):
    visited_states = []
    state_stack = [starting_state]
    extracted_connections = {}

    while len(state_stack) > 0:
        current_state = state_stack.pop()

        if current_state in visited_states:
            continue

        visited_states.append(current_state)

        if not no_tuples:
            state_key = (current_state.state_name, current_state.shorthand_state_name)
            if current_state.special_meaning != "":
                state_key = (current_state.state_name, current_state.shorthand_state_name, current_state.special_meaning)
        else:
            state_key = current_state.state_name + "|" + current_state.shorthand_state_name
            if current_state.special_meaning != "":
                state_key = current_state.state_name + "|" + current_state.shorthand_state_name + "|" + current_state.special_meaning

        extracted_connections[state_key] = []
        for next_state_name, probability in current_state.state_name_to_probability.items():
            if not no_tuples:
                extracted_connections[state_key].append((next_state_name, probability))
            else:
                extracted_connections[state_key].append(next_state_name + "|" + str(probability))
            state_stack.append(current_state.state_name_to_state[next_state_name])

    return extracted_connections

# function
#  inputs:
#   a dictionary of form dict[(name, shorter name, meaning), [(next state name a, probability a), (next state name b, probability b), etc...]]; dict[tuple[str, str, str|None], list[tuple[str, float]]]
#     or a State object
#   show plot (bool, if true, do plt.show(), otherwise don't)
#   show percentages (bool, if true, show the decimals as percentages, otherwise, just show the decimals)
#  output:
#   a matplotlib graph of the states and the connections

def maximize_plt_plot():
    plt.get_current_fig_manager().window.state("zoomed")

def plot_markov_graph(data, show_plot=True, show_as_percentages=True, use_short_hand=True, use_curved_lines=False):
    if isinstance(data, State):
        connections = extract_connection_dict(data)
    else:
        connections = data

    node_graph = nx.DiGraph()

    name_conversion_map = {state[0]: state[1] if use_short_hand else state[0] for state in connections}
    
    for source, edges in connections.items():
        for name, probability in edges:
            label = f"{probability:.2f}"
            if show_as_percentages:
                label = f"{probability * 100:.2f}%"
            node_graph.add_edge(name_conversion_map[source[0]].replace(" ", "\n"), name_conversion_map[name].replace(" ", "\n"), label=label)
    
    pos = nx.spring_layout(node_graph, seed=42, k=2, iterations=10)

    plt.figure(figsize=(8, 6))

    if not use_curved_lines:
        nx.draw(node_graph, pos, with_labels=True, node_color="white", edgecolors="black",
                node_size=6000, font_size=12, font_weight="bold", arrowsize=20)
    else:
        nx.draw(node_graph, pos, with_labels=True, node_color="white", edgecolors="black",
                node_size=6000, font_size=12, font_weight="bold", arrowsize=20, connectionstyle="arc3, rad=0.1")
    
    regular_edge_labels = {(u, v): d['label'] for u, v, d in node_graph.edges(data=True) if u != v}
    self_loop_labels = {u: d['label'] for u, v, d in node_graph.edges(data=True) if u == v}

    if not use_curved_lines:
        nx.draw_networkx_edge_labels(node_graph, pos, edge_labels=regular_edge_labels, font_color='black')
    else:
        nx.draw_networkx_edge_labels(node_graph, pos, edge_labels=regular_edge_labels, font_color='black', connectionstyle="arc3, rad=0.1")

    for node, label in self_loop_labels.items():
        x, y = pos[node]
        plt.text(x, y + 0.23, label, color='black', fontsize=10, 
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
    maximize_plt_plot()

    if show_plot:
        plt.show()

def main():
    state_connections = {
        ("Sun", "S", "Sun"): [
            ("Sun", 0.80),
            ("Cloud", 0.15),
            ("Rain", 0.05),
        ],
        ("Rain", "R", "Rain"): [
            ("Rain", 0.70),
            ("Thunderstorm", 0.20),
            ("Sun", 0.10),
        ],
        ("Thunderstorm", "TS", "Thunderstorm"): [
            ("Thunderstorm", 0.60),
            ("Rain", 0.40),
        ],
        ("Cloud", "C", "Cloud"): [
            ("Cloud", 0.45),
            ("Rain", 0.20),
            ("Thunderstorm", 0.05),
            ("Sun", 0.30),
        ],
    }

    start_state = create_connected_states(state_connections, "Sun") # String is where the Markov simulation will start
    file_path = "weather_sim.json"

    plot_markov_graph(start_state, use_curved_lines=True)
    
    # trial = Trial(start_state, special_stops=["losing"])
    sim = Simulation.from_state(start_state, special_prints=["Sun", "Rain", "Thunderstorm", "Cloud"], max_runs=5000)
    # sim = Simulation.from_file(file_path)
    sim.run_simulation(1000, 0)
    sim.create_graph()
    
    print("Mean:", sim.get_mean())
    print("SD:", sim.get_standard_deviation())

    sim.to_file(file_path, human_readable=True)

    # sim = Simulation.from_file(file_path)
    # print("Mean:", sim.get_mean())
    # print("SD:", sim.get_standard_deviation())

    # sim.create_graph()

if __name__ == "__main__":
    main()