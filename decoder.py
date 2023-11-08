from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    


        while state.buffer:
            # TODO: Write the body of this loop for part 4 
            
            # Extract features from the current state
            features = self.extractor.get_input_representation(words, pos, state)
            # Predict actions using the model
            action_probs = self.model.predict(np.array([features]))[0]
            # Create a list of possible actions sorted by probability
            possible_actions = [self.output_labels[i] for i in np.argsort(action_probs)[::-1]]

            # Find the first legal action in the sorted list
            legal_action = None
            for action in possible_actions:
                if self.is_legal_action(state, action):
                    legal_action = action
                    break

            # Perform the selected legal action
            if legal_action == "shift":
                state.shift()
            elif legal_action.startswith("left_arc"):
                _, label = legal_action.split("_")
                state.left_arc(label)
            elif legal_action.startswith("right_arc"):
                _, label = legal_action.split("_")
                state.right_arc(label)
            else:
                raise ValueError(f"Unknown action: {legal_action}")

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
    
    def is_legal_action(self, state, action):
        if action == "shift":
            return len(state.buffer) > 0
        elif action.startswith("left_arc"):
            if len(state.stack) < 2 or not state.buffer:
                return False
            _, label = action.split("_")
            return state.stack[-1] != 0  # The root node should not be the target of a left-arc
        elif action.startswith("right_arc"):
            if len(state.stack) < 2:
                return False
            _, label = action.split("_")
            return len(state.buffer) > 0

        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
