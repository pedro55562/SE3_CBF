def _detach_object(self, obj):
    ind = -1
    for i in range(len(self.attached_objects)):
        if self.attached_objects[i][0] == obj:
            ind = i

    if ind == -1:
        raise Exception("The object was not found.")

    del self.attached_objects[ind]
