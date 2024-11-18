class FillPosition:
    def __init__(self, position, item):
        self.position = position
        self.item = item

    def __call__(self, slots):
        idx = self.position - 1

        if slots[idx] is not None:
            raise ValueError(f"Position {self.position} is already filled")
        else:
            slots[idx] = self.item

        return slots


class FillNext:
    def __init__(self, item):
        self.item = item

    def __call__(self, slots):
        if None not in slots:
            raise ValueError("All recommendation slots have already been filled")

        first_empty = slots.index(None)
        slots[first_empty] = self.item

        return slots


class MultiFill:
    def __init__(self, items):
        self.items = items.copy()

    def __call__(self, slots):
        while self.items and None in slots:
            first_empty = slots.index(None)
            slots[first_empty] = self.items.pop(0)

        return slots


class MultiFillByScore:
    def __init__(self, items, scores):
        self.items = items.copy()
        self.scores = scores.copy()

    def __call__(self, slots):
        while self.items and None in slots:
            idx = self._argmax(self.scores)
            item = self.items.pop(idx)
            self.scores.pop(idx)
            first_empty = slots.index(None)
            slots[first_empty] = item

        return slots

    def _argmax(self, scores):
        return scores.index(max(scores))


SINGLE_POS_OPS = (FillPosition, FillNext)
MULTI_POS_OPS = (MultiFill, MultiFillByScore)


class LazyRecs:
    def __init__(self, num_slots):
        self.num_slots = num_slots
        self._ops = []

    @property
    def items(self):
        slots = [None] * self.num_slots

        for op in self._ops:
            if isinstance(op, FillPosition):
                slots = op(slots)

        for op in self._ops:
            if not isinstance(op, (FillPosition, MULTI_POS_OPS)):
                slots = op(slots)

        for op in self._ops:
            if isinstance(op, MULTI_POS_OPS):
                slots = op(slots)

        if None in slots:
            raise RuntimeError("Recommendations haven't been completely filled in")

        return slots

    def fill_position(self, position, item):
        if position > self.num_slots:
            raise ValueError("Slot {index} is outside the {self.num_slots} allocated slots")

        for op in self._ops:
            if isinstance(op, FillPosition):
                if op.position == position:
                    raise ValueError(f"Slot {position} has already been filled")

        self._ops.append(FillPosition(position, item))

    def fill_next(self, item):
        num_single_ops = 1 + sum([1 for op in self._ops if isinstance(op, SINGLE_POS_OPS)])
        if num_single_ops > self.num_slots:
            raise ValueError("All recommendation slots have already been filled")
        self._ops.append(FillNext(item))

    def multifill(self, items):
        self._ops.append(MultiFill(items))

    def multifill_by_score(self, items, scores):
        self._ops.append(MultiFillByScore(items, scores))
