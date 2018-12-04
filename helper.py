import torch
import numpy as np
class Helper():
    def __init__(self):
        return

    def select_action(self, model, state):
        # Samples an action according to the probability distribution induced by the model
        # Also returns the log_probability
        log_p = model.forward(torch.Tensor(state).unsqueeze(0))[0]
        action = torch.multinomial(log_p.exp(),1).item()
        return action, log_p[action]

    def run_episode(self,env, model):
        s = env.reset()
        done = False
        episode = []
        r_list = []
        while not done:
            a, log_p = self.select_action(model,s)
            s_next, r, done, _ = env.step(a)
            episode.append((s,a,log_p,r,s_next))
            s = s_next
        return episode

    def smooth(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)
