
class Task:
    def __init__(
        self,
        task_id,
        pickup,
        delivery,
        priority=1
    ):
        self.task_id = task_id
        self.pickup = pickup
        self.delivery = delivery
        self.priority = priority

        self.status = "pending"
        self.assigned_robot = None
        self.created_step = 0
        self.completed_step = None
