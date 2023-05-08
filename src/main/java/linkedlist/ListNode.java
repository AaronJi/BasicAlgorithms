package linkedlist;

public class ListNode {

    public int val;
    public ListNode next;

    //初始化创建
    ListNode() {

    }

    ListNode(int val) {
        this.val = val;
        this.next = null;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }

    // standard getters and setters
}
