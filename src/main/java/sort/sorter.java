package sort;


public class sorter {

    private int l;

    public sorter(int leng) {
        l = leng;
    }

    public static <E extends Comparable<E>> void selectionSort(E[] A) {
        System.out.println("selection sort");
        int j, k, minIndex;
        E min;
        int N = A.length;

        for (k = 0; k < N; k++) {
            min = A[k];
            minIndex = k;
            for (j = k+1; j < N; j++) {
                if (A[j].compareTo(min) < 0) {
                    min = A[j];
                    minIndex = j;
                }
            }
            A[minIndex] = A[k];
            A[k] = min;
        }

    }

    public static <E extends Comparable<E>> void insertionSort(E[] A) {
        System.out.println("insertion sort");
        int k, j;
        E tmp;
        int N = A.length;
        for (k = 1; k < N; k++) {
            tmp = A[k];
            j = k - 1;
            while ((j >= 0) && (A[j].compareTo(tmp) > 0)) {
                A[j+1] = A[j]; // move one value over one place to the right
                j--;
            }
            A[j+1] = tmp; // insert kth value in correct place relative to previous values
        }
    }

    public static <E extends Comparable<E>> void insertionSort(E[] A, int low, int high) {
        System.out.println("insertion sort");
        int k, j;
        E tmp;
        int N = A.length;
        for (k = 1; k < N; k++) {
            tmp = A[k];
            j = k - 1;
            while ((j >= 0) && (A[j].compareTo(tmp) > 0)) {
                A[j+1] = A[j]; // move one value over one place to the right
                j--;
            }
            A[j+1] = tmp; // insert kth value in correct place relative to previous values
        }
    }

    /*
    * 1. divide the array into two halvs
    * 2. recursively, sort the left half
    * 3. recursively, sort the right half
    * 4. merge the two sorted halves
    * */
    public static <E extends Comparable<E>> void mergeSort(E[] A) {
        mergeAux(A, 0, A.length - 1); // call the aux. function to do all the work
    }
    private static <E extends Comparable<E>> void mergeAux(E[] A, int low, int high) {
        // base case
        if (low == high) return;
        // recursive case
        // step 1: find the middle of the array (conceptually, divide it in half)
        int mid = (low + high) / 2;
        // step 2 and 3: sort the 2 halves of A
        mergeAux(A, low, mid);
        mergeAux(A, mid+1, high);
        // step 4: merge sorted halves ino an auxiliary array
        E[] tmp = (E[])(new Comparable[high-low+1]);
        int left = low; // index into left half
        int right = mid + 1; // index into right half
        int pos = 0; // index into tmp
        while ((left <= mid) && (right <= high)) {
            // choose the smaller of the two halves "pointed to" by left, right
            // copy that value into tmp[pos]
            // increment either left or right as appropriate
            // increment pos

        }
        // when one of the two sorted halves has "run out" of values, but there are still some in the other half, copy all the remaining values to tmp
        // Note: only 1 of the next 2 loops will actually execute
        while (left <= mid) {

        }
        while (right <= high) {

        }
        System.arraycopy(tmp, 0, A, low, tmp.length);
    }

    /*
    * start by partitioning the array: putting all small values in the left half and putting all large values in the right half.
    * no need for a combine step.
    * 1. choose a privot value
    * 2. partition the array
    * 3. recursively, sort the values less (or equal to) than the pivot
    * 4. recursively, sort the values greater than (or equal to) the pivot
    * */
    public static <E extends Comparable<E>> void quickSort(E[] A) {
        quickAux(A, 0, A.length-1);
    }
    private static <E extends Comparable<E>> void quickAux(E[] A, int low, int high) {
        if (high - low < 4) insertionSort(A, low, high);
        else {
            int right = partition(A, low, high);
            quickAux(A, low, right);
            quickAux(A, right + 2, high);
        }
    }
    private static <E extends Comparable<E>> int partition(E[] A, int low, int high) {
        // precondition: A.length > 3
        E pivot = medianOfThree(A, low, high);
        int left = low + 1;
        int right = high -2;
        while (left <= right) {
            while(A[left].compareTo(pivot) < 0) left++;
            while(A[right].compareTo(pivot) > 0) right--;
            if (left <= right) {
                swap(A, left, right);
                left++;
                right--;
            }

        }
        swap(A, right+1, high-1);
        return right;
    }


    // TODO
    private static <E extends Comparable<E>> E medianOfThree(E[] A, int low, int high) {
        return A[0];
    }

    // TODO
    private static <E extends Comparable<E>> void swap(E[] A, int left, int right) {
        return;
    }


}