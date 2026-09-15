/* mmapprobe — what one weight sweep costs from an mmap'd f32 file vs the heap.
 * Shape: a stage-4 molequla organism, 4 834 408 parameters, float32.
 * One sweep = a matvec over every parameter, which is the weight traffic of one
 * forward pass. Four arms: heap, mmap warm, mmap after MADV_DONTNEED (page cache
 * still holds the file), mmap after POSIX_FADV_DONTNEED (cache dropped too).  */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define NPARAM 4834408L
#define COLS   224
static const long ROWS = NPARAM / COLS;   /* 21582 rows, 4834368 elements used */

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1000.0 + t.tv_nsec/1e6; }

static long procKB(const char *key){
    FILE *f=fopen("/proc/self/status","r"); if(!f) return 0; char l[256]; long v=0;
    while(fgets(l,sizeof l,f)) if(!strncmp(l,key,strlen(key))){ sscanf(l+strlen(key)+1,"%ld",&v); break; }
    fclose(f); return v; }

static long faults(int major){
    FILE *f=fopen("/proc/self/stat","r"); if(!f) return 0;
    long min=0,maj=0; /* fields 10 minflt, 12 majflt */
    /* skip pid and comm */
    int c; while((c=fgetc(f))!=')' && c!=EOF); 
    long dummy; char st;
    if(fscanf(f," %c %ld %ld %ld %ld %ld %ld %ld %ld %ld",&st,&dummy,&dummy,&dummy,&dummy,&dummy,&dummy,&min,&dummy,&maj)!=10){fclose(f);return 0;}
    fclose(f); return major?maj:min; }

static double sweep(const float *w, const float *x, float *y){
    double t0=now_ms();
    for(long r=0;r<ROWS;r++){
        const float *row=w+r*COLS; float acc=0.f;
        for(int j=0;j<COLS;j++) acc += row[j]*x[j];
        y[r]=acc;
    }
    return now_ms()-t0; }

int main(int argc,char**argv){
    const char *path = argc>1?argv[1]:"weights.f32";
    int reps = argc>2?atoi(argv[2]):5;
    size_t bytes = (size_t)NPARAM*sizeof(float);

    /* build the file once */
    struct stat st;
    if(stat(path,&st)!=0 || (size_t)st.st_size!=bytes){
        float *buf=malloc(bytes); if(!buf){perror("malloc");return 1;}
        for(long i=0;i<NPARAM;i++) buf[i]=(float)((i%1000)-500)/1000.0f;
        int fd=open(path,O_CREAT|O_TRUNC|O_WRONLY,0644); if(fd<0){perror("open");return 1;}
        if(write(fd,buf,bytes)!=(ssize_t)bytes){perror("write");return 1;}
        fsync(fd); close(fd); free(buf);
    }
    printf("file %s  %zu bytes (%.2f MB), rows=%ld cols=%d, reps=%d\n",
           path,bytes,bytes/1e6,ROWS,COLS,reps);

    float *x=malloc(COLS*sizeof(float)); for(int j=0;j<COLS;j++) x[j]=0.01f*j;
    float *y=malloc(ROWS*sizeof(float));
    long rss0=procKB("VmRSS");

    /* ---- arm 1: heap ---- */
    {
        int fd=open(path,O_RDONLY);
        float *w=malloc(bytes);
        double t0=now_ms();
        ssize_t got=0,n; while((n=read(fd,(char*)w+got,bytes-got))>0) got+=n;
        double load=now_ms()-t0; close(fd);
        double first=sweep(w,x,y);
        double best=1e18,sum=0; for(int r=0;r<reps;r++){double d=sweep(w,x,y); sum+=d; if(d<best)best=d;}
        printf("heap        : load %7.1f ms | first sweep %6.2f ms | mean %6.2f ms | best %6.2f ms | VmRSS %ld kB\n",
               load,first,sum/reps,best,procKB("VmRSS"));
        free(w);
    }
    printf("rss after heap arm freed: %ld kB (baseline %ld)\n",procKB("VmRSS"),rss0);

    /* ---- arms 2-4: mmap ---- */
    int fd=open(path,O_RDONLY); if(fd<0){perror("open");return 1;}
    float *w=mmap(NULL,bytes,PROT_READ,MAP_PRIVATE,fd,0);
    if(w==MAP_FAILED){perror("mmap");return 1;}

    long mn0=faults(0),mj0=faults(1);
    double firstm=sweep(w,x,y);
    printf("mmap first  : %6.2f ms (fault-in from page cache) | minflt +%ld majflt +%ld | VmRSS %ld kB\n",
           firstm,faults(0)-mn0,faults(1)-mj0,procKB("VmRSS"));

    {   double best=1e18,sum=0; for(int r=0;r<reps;r++){double d=sweep(w,x,y); sum+=d; if(d<best)best=d;}
        printf("mmap warm   : mean %6.2f ms | best %6.2f ms | VmRSS %ld kB\n",sum/reps,best,procKB("VmRSS")); }

    {   double sum=0; long mn=0,mj=0;
        for(int r=0;r<reps;r++){
            madvise(w,bytes,MADV_DONTNEED);
            long a=faults(0),b=faults(1);
            sum+=sweep(w,x,y); mn+=faults(0)-a; mj+=faults(1)-b;
        }
        printf("mmap evicted: mean %6.2f ms | minflt/sweep %ld majflt/sweep %ld | VmRSS %ld kB  (pages dropped from the process, file still cached)\n",
               sum/reps,mn/reps,mj/reps,procKB("VmRSS")); }

    {   double sum=0; long mj=0; int n=reps<3?reps:3;
        for(int r=0;r<n;r++){
            madvise(w,bytes,MADV_DONTNEED);
            posix_fadvise(fd,0,bytes,POSIX_FADV_DONTNEED);
            long b=faults(1);
            sum+=sweep(w,x,y); mj+=faults(1)-b;
        }
        printf("mmap cold   : mean %6.2f ms over %d | majflt/sweep %ld | VmRSS %ld kB  (page cache dropped too: read from flash)\n",
               sum/n,n,mj/n,procKB("VmRSS")); }

    printf("VmHWM %ld kB\n",procKB("VmHWM"));
    munmap(w,bytes); close(fd);
    return 0;
}
