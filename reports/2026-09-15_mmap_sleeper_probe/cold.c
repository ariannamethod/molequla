/* cold.c — two questions the first probe could not answer:
 *  (a) does the resident cost of an mmap'd sleeper actually go away when it is
 *      not speaking (MADV_DONTNEED, no sweep afterwards)?
 *  (b) what does a sweep cost when the file is not in the page cache at all —
 *      which needs the mapping gone before POSIX_FADV_DONTNEED can evict it. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>
#define NPARAM 4834408L
#define COLS 224
static const long ROWS = NPARAM/COLS;
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static long procKB(const char*k){FILE*f=fopen("/proc/self/status","r");char l[256];long v=0;
 while(fgets(l,sizeof l,f))if(!strncmp(l,k,strlen(k))){sscanf(l+strlen(k)+1,"%ld",&v);break;}fclose(f);return v;}
static long majflt(void){FILE*f=fopen("/proc/self/stat","r");int c;while((c=fgetc(f))!=')'&&c!=EOF);
 char st;long d,mj=0;if(fscanf(f," %c %ld %ld %ld %ld %ld %ld %ld %ld %ld",&st,&d,&d,&d,&d,&d,&d,&d,&d,&mj)!=10)mj=0;fclose(f);return mj;}
static double sweep(const float*w,const float*x,float*y){double t0=now_ms();
 for(long r=0;r<ROWS;r++){const float*row=w+r*COLS;float a=0.f;for(int j=0;j<COLS;j++)a+=row[j]*x[j];y[r]=a;}
 return now_ms()-t0;}
int main(int argc,char**argv){
 const char*path=argv[1]; size_t bytes=(size_t)NPARAM*sizeof(float);
 float*x=malloc(COLS*4);for(int j=0;j<COLS;j++)x[j]=0.01f*j; float*y=malloc(ROWS*4);
 printf("baseline VmRSS %ld kB\n",procKB("VmRSS"));
 int fd=open(path,O_RDONLY); float*w=mmap(NULL,bytes,PROT_READ,MAP_PRIVATE,fd,0);
 double t=sweep(w,x,y); printf("after one sweep      : %6.2f ms | VmRSS %ld kB\n",t,procKB("VmRSS"));
 madvise(w,bytes,MADV_DONTNEED);
 printf("after MADV_DONTNEED  : (no sweep)    | VmRSS %ld kB  <- the sleeper's idle cost\n",procKB("VmRSS"));
 t=sweep(w,x,y); printf("speaking again       : %6.2f ms | VmRSS %ld kB\n",t,procKB("VmRSS"));
 /* cold from flash: drop the mapping, then the cache, then map again */
 munmap(w,bytes);
 posix_fadvise(fd,0,bytes,POSIX_FADV_DONTNEED);
 long m0=majflt();
 w=mmap(NULL,bytes,PROT_READ,MAP_PRIVATE,fd,0);
 t=sweep(w,x,y);
 printf("cold from flash      : %6.2f ms | majflt +%ld | VmRSS %ld kB\n",t,majflt()-m0,procKB("VmRSS"));
 t=sweep(w,x,y); printf("warm again           : %6.2f ms | VmRSS %ld kB\n",t,procKB("VmRSS"));
 printf("VmHWM %ld kB\n",procKB("VmHWM"));
 munmap(w,bytes);close(fd);return 0;}
