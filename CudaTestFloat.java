import static jcuda.driver.JCudaDriver.*;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

//https://github.com/jcuda/jcuda-samples

public class CudaTestFloat {
    public static float run(int n, boolean memory) {
        long t1 = 0;
        long time = 0;
        JCudaDriver.setExceptionsEnabled(true);

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Load the ptx file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, "module/vectoradd_float.ptx");

        // Obtain a function pointer to the "add" function.
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "add");

        // Allocate and fill the host input data
        float hostInputA[] = new float[n];
        float hostInputB[] = new float[n];
        for(int i = 0; i < n; i++) {
            hostInputA[i] = (float) i;
            hostInputB[i] = (float) i;
        }

        if(memory) t1 = System.nanoTime();

        // Allocate the device input data, and copy the
        // host input data to the device
        CUdeviceptr deviceInputA = new CUdeviceptr();
        cuMemAlloc(deviceInputA, n * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceInputA, Pointer.to(hostInputA), n * Sizeof.FLOAT);
        CUdeviceptr deviceInputB = new CUdeviceptr();
        cuMemAlloc(deviceInputB, n * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceInputB, Pointer.to(hostInputB), n * Sizeof.FLOAT);

        // Allocate device output memory
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, n * Sizeof.FLOAT);

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{n}),
                Pointer.to(deviceInputA),
                Pointer.to(deviceInputB),
                Pointer.to(deviceOutput)
        );

        if(!memory) t1 = System.nanoTime();
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

        if(!memory) time = System.nanoTime() - t1;

        // Allocate host output memory and copy the device output
        // to the host.
        float hostOutput[] = new float[n];
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, n * Sizeof.FLOAT);

        if(memory) time = System.nanoTime() - t1;

        // Clean up.
        cuMemFree(deviceInputA);
        cuMemFree(deviceInputB);
        cuMemFree(deviceOutput);

        //Output
        //for (int i = 0; i < n; i++) System.out.println(hostInputA[i] + " + " + hostInputB[i] + " = " + hostOutput[i]);

        return(((float)time)/1000000);

    }
}