

import java.util.Arrays;
import java.util.Vector;
import java.util.List;

import sort.sorter;
import search.searcher;
import utils.other;
import automator.Solution;

public class Main {

    public static void main(String[] args) {
        System.out.println("Hello World!");

        //test_sort();
        //test();
        //test_other();
        //test_int();
        //test_string();

        //other.test_string();
        //other.test_getKthElement();

        //System.out.println(other.lengthOfLongestSubstring("adasrrwwdwr"));
        //System.out.println(other.findSubstring("barfoothefoobarman", new String[]{"foo", "bar"}));
        //System.out.println(other.firstMissingPositive(new int[]{4, 1, 2, 5, 4}));
        System.out.println(other.romanToInt("MDCCCLXXXIV"));
        System.out.println(Arrays.asList(2,3,1,4));
    }

    private static void test_other() {
        int[] A = {5, 2, 3, 1, 4};
        int target =6;
        other theOther = new other();
        int[] result = theOther.twoSum(A, target);
        System.out.println(Arrays.toString(result));
    }

    private static void test_int() {
        other theOther = new other();
        int x = 12321;
        boolean is_p = theOther.isPalindrome(x);
        System.out.println(is_p);
    }

    private static void test_string() {
        Solution sol = new Solution();    
        String s = "256";
        int result = sol.myAtoi(s);
        System.out.println(result);
    }

    private static void  test_sort() {
        Integer[] A = {5, 2, 3, 1, 4};
        sorter theSorter = new sorter(A.length);
        //theSorter.selectionSort(A);
        theSorter.insertionSort(A);
        System.out.println(Arrays.toString(A));
    }

    private static void test() {
        String valuet = "33747.0,1529.0,214.0,6.0,977.0,4.0,25.0,100.0";

        System.out.println(valuet.split(",").length);
        double ipv_target = Double.parseDouble(valuet.split(",")[6]);
        double pv_target = Double.parseDouble(valuet.split(",")[7]);
        double ipv_acc = Double.parseDouble(valuet.split(",")[3]);
        double pv_acc = Double.parseDouble(valuet.split(",")[2]);

        System.out.println(ipv_target);
        System.out.println(pv_target);
        System.out.println(ipv_acc);
        System.out.println(pv_acc);

    }


    int findDuplicate1(Vector<Integer> nums) {
        Vector<Integer> hash = new Vector<Integer>(nums.size() + 1);
        for (Integer i: nums){
            if (hash.get(i) == 0)
                hash.set(i, 1);
            else
                return i;
        }
        return -1;
    }



    /*
    int LIS(int[] array)
    {
    int *LIS = new int[array.Length];
        for(int i = 0; i < array.Length; i++)
        {
            LIS[i] = 1;          //初始化默认的长度
            for(int j = 0; j < i; j++)        //前面最长的序列
            {
                if(array[i] > array[j] && LIS[j] + 1 > LIS[i])
                {
                    LIS[i] = LIS[j] + 1;
                }
            }
        }
        return Max(LIS);      //取LIS的最大值
    }

    // 楼梯走法数
    #include <iostream>
    using namespace std;
    int dp[ 10001 ] = { 0 };
    int main()
    {
        int num;
        cin >> num;
        dp[ 1 ] = 1;
        dp[ 2 ] = 2;
        dp[ 3 ] = 4;
        for(int i = 4; i <= num; i++){
            dp[ i ] = dp[ i - 1 ]+ dp[ i - 2 ] + dp[ i - 3 ];
        }
        cout<< dp[ num ] <<endl;
        return 0;
    }
    */
}
