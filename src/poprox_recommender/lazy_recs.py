from copy import deepcopy


class LazyRecs:
    def __init__(self, num_slots):
        self.num_slots = num_slots
        self._ops = []

    def fill_position(self, position, item):
        if position > self.num_slots:
            raise ValueError("Slot {index} is outside the {self.num_slots} allocated slots")

        for op in self._ops:
            if isinstance(op, FillPosition):
                if op.position == position:
                    raise ValueError(f"Slot {position} has already been filled")

        self._ops.append(FillPosition(position, item))

    def fill_next(self, item):
        num_single_ops = 1 + sum([1 for op in self._ops if isinstance(op, SinglePositionOp)])
        if num_single_ops > self.num_slots:
            raise ValueError("All recommendation slots have already been filled")
        self._ops.append(FillNext(item))

    def multifill(self, items):
        self._ops.append(MultiFill(items))

    def multifill_by_score(self, items, scores):
        self._ops.append(MultiFillByScore(items, scores))

    def constrain_above_position(self, position, items, scores):
        constraint_ops = [op for op in self._ops if isinstance(op, ConstrainAbovePosition)]
        constrained_items_above = sum([len(op.items) for op in constraint_ops if op.position <= position])
        if len(items) + constrained_items_above > position:
            raise ValueError(f"Too many items to constrain above position {position}")
        self._ops.append(ConstrainAbovePosition(position, items, scores))

    @property
    def items(self):
        slots = [None] * self.num_slots
        scores = [None] * self.num_slots

        for op in self._ops:
            if isinstance(op, MultiPositionOp):
                # Filter fixed position and constrained items out of multifill pools
                filtered_op = self._filter_multifill_pool(op)
                slots, scores = filtered_op(slots, scores)

        # Execute constraints from the bottom of the list upward,
        # adjusting constraints to account for later inserts
        for op in sorted(self._constraint_ops, key=lambda o: o.position, reverse=True):
            adjusted_op = self._adjust_constraint_position(op, self._ops)
            slots, scores = adjusted_op(slots, scores)

        for op in self._ops:
            if isinstance(op, (FillNext,)):
                slots, scores = op(slots, scores)

        for op in self._ops:
            if isinstance(op, FillPosition):
                slots, scores = op(slots, scores)

        slots = slots[: self.num_slots]

        if None in slots:
            raise RuntimeError("Recommendations haven't been completely filled in")

        return slots

    def _filter_multifill_pool(self, op):
        filtered_op = deepcopy(op)
        for item in self._fixed_pos_items + self._constrained_pos_items:
            try:
                idx = filtered_op.items.index(item)
                filtered_op.items.pop(idx)
                if hasattr(filtered_op, "scores"):
                    filtered_op.scores.pop(idx)
            except ValueError:
                pass
        return filtered_op

    def _adjust_constraint_position(self, op, other_ops):
        fixed_pos_above = sum(
            [1 for fill_op in other_ops if isinstance(fill_op, FillPosition) and fill_op.position < op.position]
        )
        constrained_above = sum(
            [
                len(constraint_op.items)
                for constraint_op in other_ops
                if isinstance(constraint_op, ConstrainAbovePosition) and constraint_op.position < op.position
            ]
        )
        adjusted_op = deepcopy(op)
        adjusted_op.position -= fixed_pos_above
        adjusted_op.position -= constrained_above
        return adjusted_op

    @property
    def _fixed_pos_items(self):
        fixed_pos_items = []
        for op in self._ops:
            if isinstance(op, FillPosition):
                fixed_pos_items.append(op.item)
        return fixed_pos_items

    @property
    def _constrained_pos_items(self):
        constrained_items = []
        for op in self._ops:
            if isinstance(op, ConstrainAbovePosition):
                constrained_items.extend(op.items)
        return constrained_items

    @property
    def _constraint_ops(self):
        constraint_ops = []
        for op in self._ops:
            if isinstance(op, (ConstrainAbovePosition,)):
                constraint_ops.append(op)
        return constraint_ops


class SinglePositionOp:
    pass


class MultiPositionOp:
    pass


class ConstraintOp:
    pass


class FillPosition(SinglePositionOp):
    def __init__(self, position, item):
        self.position = position
        self.item = item

    def __call__(self, slots, scores):
        idx = self.position - 1

        items = slots[:idx] + [self.item] + slots[idx:]
        scores = scores[:idx] + [None] + scores[idx:]

        return items, scores


class FillNext(SinglePositionOp):
    def __init__(self, item):
        self.item = item

    def __call__(self, slots, scores):
        if None not in slots:
            raise ValueError("All recommendation slots have already been filled")

        first_empty = slots.index(None)
        slots[first_empty] = self.item
        scores[first_empty] = None

        return slots, scores


class MultiFill(MultiPositionOp):
    def __init__(self, items):
        self.items = items.copy()

    def __call__(self, slots, scores):
        while self.items and None in slots:
            first_empty = slots.index(None)
            slots[first_empty] = self.items.pop(0)
            scores[first_empty] = None

        return slots, scores


class MultiFillByScore(MultiPositionOp):
    def __init__(self, items, scores):
        self.items = items.copy()
        self.scores = scores.copy()

    def __call__(self, slots, scores):
        while self.items and None in slots:
            idx = self._argmax(self.scores)
            item = self.items.pop(idx)
            score = self.scores.pop(idx)

            first_empty = slots.index(None)
            slots[first_empty] = item
            scores[first_empty] = score

        return slots, scores

    def _argmax(self, scores):
        return scores.index(max(scores))


class ConstrainAbovePosition(ConstraintOp):
    def __init__(self, position, items, scores):
        self.position = position
        self.items = items
        self.scores = scores

    def __call__(self, slots, scores):
        total_slots = len(slots)

        top = self.position - len(self.items)
        bottom = self.position

        items = (slots[:top] + self.items + slots[top:])[:total_slots]
        scores = (scores[:top] + self.scores + scores[top:])[:total_slots]

        # Bubble constrained items upward by score
        for i in range(top, bottom):
            for j in range(i, 0, -1):
                if scores[i] > scores[j]:
                    items[i], items[j] = items[j], items[i]
                    scores[i], scores[j] = scores[j], scores[i]

        return items, scores
