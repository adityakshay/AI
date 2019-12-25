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