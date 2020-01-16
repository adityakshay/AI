
from sample_players import DataPlayer
import logging
import pickle
import random
from isolation import isolation
from collections import defaultdict
import time, math, random



class montecarlotreesearchnode():
    """
    Monte Carlo Tree Search node class
    """
    
    def __init__(self, state: isolation, action=None, parent=None):
        '''
        @param state: Game state included in this node
        @param parent: Parent node for current node  
        '''
        
        self.state=state
        self.parent=parent
        self.children=[]
        self.action=action
        self.q=0
        self.n=0
        self._results = defaultdict(int)
        self.untried_actions=state.actions()
        
        
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param = 1.4):
        choices_weights = [(c.q / (c.n)) + c_param * math.sqrt(2*math.log(self.n) / (c.n)) \
                           for c in self.children]
        return self.children[choices_weights.index(max(choices_weights))]
    
    def max_q_child(self):
        choices_weights = [c.q for c in self.children]
        return self.children[choices_weights.index(max(choices_weights))]
    
    def rollout_policy(self,possible_moves):
        return random.choice(possible_moves)
    '''
    @property
    def untried_actions(self):
        if not hasattr(self, '_untried_actions'):
            self._untried_actions = self.state.actions()
        print("untried_actions: "+str(self._untried_actions))
        return self._untried_actions    
        
    @property
    def q(self):
        wins = self._results[self.state.player()]
        loses = self._results[1-self.state.player()]
        #print("q wins "+str(wins)+", losses "+str(loses)+", value "+str(wins-loses))
        return wins - loses    
    
    @property
    def n(self):
        #print("n "+str(self._number_of_visits))
        return self._number_of_visits
    '''
    
    def expand(self):
        res_action=random.choice(self.untried_actions)
        self.untried_actions.remove(res_action)
        next_state=self.state.result(res_action)
        child_node = montecarlotreesearchnode(next_state, action=res_action, parent = self)
        self.children.append(child_node)
        return child_node
    
    def is_terminal_node(self):
        return self.state.terminal_test()
    
    def rollout(self,player_id):
        current_rollout_state = self.state
        while not current_rollout_state.terminal_test():
            current_rollout_state = current_rollout_state.result(self.rollout_policy(current_rollout_state.actions()))
        if current_rollout_state.utility(1-current_rollout_state.player())==float('inf'):  
            return 1.
        elif current_rollout_state.utility(1-current_rollout_state.player())==float('-inf'):
            return -1.
        else:
            return 0
        
    def backpropagate(self, result):
        self.n += 1.
        self.q += result
        result=-result
        if self.parent:
            self.parent.backpropagate(result)
    
    
            
class montecarlotreesearch():
    '''
    Perform Monte Carlo Tree Search
    '''
    
    def __init__(self, node: montecarlotreesearchnode):
        self.root = node
        self.player_id=self.root.state.player()
        self.node_no=1
        
    def best_action(self, simulations_time):
        start=time.time()
        allowed_time=math.ceil(simulations_time*0.75)
        while (time.time()-start)*1000<=allowed_time:
            v = self.tree_policy()
            reward = v.rollout(self.player_id)
            v.backpropagate(reward)
        res_node=self.root.best_child(c_param = 0.5)
        res_act=res_node.action
        return res_act
    
    def tree_policy(self):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                new_node=current_node.expand()
                self.node_no+=1
                return new_node
            else:
                current_node = current_node.best_child(c_param=0.5)
        return current_node

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        
        #self.queue.put(random.choice(state.actions()))
        if state.terminal_test() or state.ply_count < 2 :
            self.queue.put(random.choice(state.actions()))
        else:
            root = montecarlotreesearchnode(state = state)
            mcts = montecarlotreesearch(root)
            bact = mcts.best_action(100)
            self.queue.put(bact)

