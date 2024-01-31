import copy
from typing import List, Any
import queue


class Observable:

    _observable: Any
    _mailboxes: List[queue.Queue]

    def __init__(self, observable: Any):
        super(Observable, self).__setattr__('_observable', observable)
        super(Observable, self).__setattr__('_mailboxes', [])

    def subscribe(self, mailboxes: List[queue.Queue]):
        self._mailboxes += mailboxes

    def __setattr__(self, key, value):
        self._observable.key = value

        self.notify()

    def __getattr__(self, key):
        print('deded',key,self._observable.key,getattr(self._observable, key))
        return self._observable.key


    def notify(self):
        observable_class_name: str = type(self._observable).__name__
        observable_copy = copy.copy(self._observable)
        message: dict = {
            'type': observable_class_name,
            observable_class_name: observable_copy
        }
        for mailbox in self._mailboxes:
            mailbox.put(message)
