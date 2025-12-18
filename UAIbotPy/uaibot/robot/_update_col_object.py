# Update collision object pose (for drawing purpose only)
def _update_col_object(self, time, mode):
    htm = self.fkm(axis="dh", mode=mode)

    for i in range(len(self.links)):
        for j in range(len(self.links[i].col_objects)):
            self.links[i].col_objects[j][0].add_ani_frame(time, htm[i] * self.links[i].col_objects[j][1])
