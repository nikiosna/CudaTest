public class CudaTest {
    public static void main(String[] args) {
        String test = null;
        int n = 80000000;
        boolean memory = false;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-d": {
                    test = "double";
                    break;
                }
                case "--double": {
                    test = "double";
                    break;
                }
                case "-f": {
                    test = "float";
                    break;
                }
                case "--float": {
                    test = "float";
                    break;
                }
                case "-m": {
                    memory = true;
                    break;
                }
                case "--memory": {
                    memory = true;
                    break;
                }
                case "-n": {
                    n = Integer.valueOf(args[i+1]);
                    break;
                }
                case "--number": {
                    n = Integer.valueOf(args[i+1]);
                    break;
                }
                default: {
                    if(!(args[i-1].equals("-n") || args[i-1].equals("--number"))) {
                        printUsage();
                        System.exit(-1);
                    }
                }
            }
        }
        if(args.length==0 || test==null) {
            printUsage();
            System.exit(-1);
        }

        //CUDA test
        float gpu_time = -1;
        if(test.equals("float")) {
            gpu_time = CudaTestFloat.run(n, memory);
        } else if (test.equals("double")) {
            gpu_time = CudaTestDouble.run(n, memory);
        }

        //CPU test
        long t1 = 0;
        if(memory) t1 = System.nanoTime();
        float A[] = new float[n];
        float B[] = new float[n];
        float C[] = new float[n];
        for(int i = 0; i < n; i++) {
            A[i] = (float) i;
            B[i] = (float) i;
        }
        if(!memory) t1 = System.nanoTime();
        for(int i = 0; i < n; i++) {
            C[i] = A[i] + B[i];
        }
        long time = System.nanoTime() - t1;
        float cpu_time = ((float)time)/1000000;

        System.out.println("Time needed to add " + n + " numbers");
        System.out.println("CUDA / GPU: " + gpu_time + " ms");
        System.out.println("CPU: " + cpu_time + " ms");

    }

    static void printUsage() {
        System.out.println("Syntax:");
        System.out.println("Necessary --double or --float");
        System.out.println("Additional --number [int] (default is 10000000)");
        System.out.println("           --memory  include the memory copy time");
    }
}
