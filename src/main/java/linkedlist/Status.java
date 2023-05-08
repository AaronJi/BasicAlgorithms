package linkedlist;

class Status implements Comparable<Status> {
    int val;
    ListNode ptr;

    Status(int val, ListNode ptr) {
        this.val = val;
        this.ptr = ptr;
    }

    public int compareTo(Status status2) {
        return this.val - status2.val;
    }
}
