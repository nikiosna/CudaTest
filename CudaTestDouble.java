import static jcuda.driver.JCudaDriver.*;
import jcuda.*;
import jcuda.driver.*;

//https://github.com/jcuda/jcuda-samples

public class CudaTestDouble {
    public static float run(int n) {
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Load the ptx file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, "module/vectoradd_double.ptx");

        // Obtain a function pointer to the "add" function.
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "add");

        // Allocate and fill the host input data
        double hostInputA[] = new double[n];
        double hostInputB[] = new double[n];
        for(int i = 0; i < n; i++) {
            hostInputA[i] = (double) i;
            hostInputB[i] = (double) i;
        }

        // Allocate the device input data, and copy the
        // host input data to the device
        CUdeviceptr deviceInputA = new CUdeviceptr();
        cuMemAlloc(deviceInputA, n * Sizeof.DOUBLE);
        cuMemcpyHtoD(deviceInputA, Pointer.to(hostInputA), n * Sizeof.DOUBLE);
        CUdeviceptr deviceInputB = new CUdeviceptr();
        cuMemAlloc(deviceInputB, n * Sizeof.DOUBLE);
        cuMemcpyHtoD(deviceInputB, Pointer.to(hostInputB), n * Sizeof.DOUBLE);

        // Allocate device output memory
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, n * Sizeof.DOUBLE);

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{n}),
                Pointer.to(deviceInputA),
                Pointer.to(deviceInputB),
                Pointer.to(deviceOutput)
        );

        long t1 = System.nanoTime();
        // Call the kernel function.
        int blockSizeX = 256;
        int gridSizeX = (int)Math.ceil((double)n / blockSizeX);
        cuLaunchKernel(function,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        long time = System.nanoTime() - t1;

        // Allocate host output memory and copy the device output
        // to the host.
        double hostOutput[] = new double[n];
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, n * Sizeof.DOUBLE);

        // Clean up.
        cuMemFree(deviceInputA);
        cuMemFree(deviceInputB);
        cuMemFree(deviceOutput);

        //Output
        //for (int i = 0; i < n; i++) System.out.println(hostInputA[i] + " + " + hostInputB[i] + " = " + hostOutput[i]);

        return(((float)time)/1000000);

    }
}