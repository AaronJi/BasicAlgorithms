package recursion;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

import linkedlist.ListNode;

/*
 * 17 Medium
 * 电话号码的字母组合
 * m: 对应3个字母的数字个数
 * n: 对应4个字母的数字个数
 * 时间复杂度：O(3^m * 4^n)
 * 空间复杂度：O(m + n)
 * */
public class Recursor {
    public List<String> letterCombinations(String digits) {
        List<String> combinations = new ArrayList<String>();
        if (digits.length() == 0) {
            return combinations;
        }
        Map<Character, String> phoneMap = new HashMap<Character, String>() {{
            put('2', "abc");
            put('3', "def");
            put('4', "ghi");
            put('5', "jkl");
            put('6', "mno");
            put('7', "pqrs");
            put('8', "tuv");
            put('9', "wxyz");
        }};
        backtrack(combinations, phoneMap, digits, 0, new StringBuffer());
        return combinations;
    }

    public void backtrack(List<String> combinations, Map<Character, String> phoneMap, String digits, int index, StringBuffer combination) {
        if (index == digits.length()) {
            combinations.add(combination.toString());
        } else {
            char digit = digits.charAt(index);
            String letters = phoneMap.get(digit);
            int lettersCount = letters.length();
            for (int i = 0; i < lettersCount; i++) {
                combination.append(letters.charAt(i));
                backtrack(combinations, phoneMap, digits, index + 1, combination);
                combination.deleteCharAt(index);
            }
        }
    }


    /*
     * 21 Easy
     * 合并两个有序链表
     * 时间复杂度：O(n + m)
     * 空间复杂度：O(n + m)
     * */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        } else if (l2 == null) {
            return l1;
        } else if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }

    /*
     * 2 括号生成
     * 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
     * 时间复杂度：O()
     * 空间复杂度：O()
     * */

    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        if(n <= 0){
            return res;
        }
        getParenthesis(res,"",n,n);
        return res;
    }

    private void getParenthesis(List<String> res, String str,int left, int right) {
        if(left == 0 && right == 0 ){
            res.add(str);
            return;
        }
        if(left == right){
            //剩余左右括号数相等，下一个只能用左括号
            getParenthesis(res,str+"(",left-1,right);
        }else if(left < right){
            //剩余左括号小于右括号，下一个可以用左括号也可以用右括号
            if(left > 0){
                getParenthesis(res,str+"(",left-1,right);
            }
            getParenthesis(res, str+")",left,right-1);
        }
    }
}
