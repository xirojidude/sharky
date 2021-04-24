
Shader "Skybox/OpticalDestruction"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _SunDir ("Sun Dir", Vector) = (-.11,.07,0.99,0) 
        _XYZPos ("XYZ Offset", Vector) = (0, 15, -.25 ,0) 
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            uniform sampler2D _MainTex; 
            float4 _SunDir,_XYZPos;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 worldPos : TEXCOORD1;
                float4 vertex : SV_POSITION;   //pos
                float4 screenPos : TEXCOORD2;
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                o.worldPos = mul(unity_ObjectToWorld, v2.vertex);
                o.screenPos  = ComputeScreenPos(o.vertex);
                //o.screenPos.z = -(mul(UNITY_MATRIX_MV, vertex).z * _ProjectionParams.w);     // old calculation, as I used the depth buffer comparision for min max ray march. 

                return o;
            }


//scene1: https://www.shadertoy.com/view/MlfczH
//self 5: https://www.shadertoy.com/view/4tscR8
//Optical-Circuit optical circuit scene 5 deconstruction b

/*

not my code, just deconstructing it:

www.pouet.net/prod.php?which=65125
https://www.youtube.com/watch?v=ISkIB4w6v6I

The [optical circuit demo] video source code once appeared on glslsandbox.com
... with very nondesctiptic labels, 
... only using single letter names for functions and vars..

It is fracal code golf overkill in [0..6] scenes.
This is a deconstruction of scene 5. , not the whole demo.
Un-used functions (only used in other scenes) are removed;
scene-specific branches are set to 1, or removed 
... (multiplying by *0. or adding -0 iterations)
... all changes are annotated.

This may run slightly faster due to removing all schene-specific branching
Most of that modifies iteration count (between scenes, which are static per shader)
The [smart re-use of schene specific branches and modifiers] is what makes this a 4k demo.
... at a cost of running slightly slower, by summing up scene-modifiers.
*/


//#define scene 5
#define timeOffset 23.984083

//scene5 skips all the black fracal code.
//and it replaces it with +50% more iterations over the 2 main loops

const float pi=acos(-1.);//3.14
const float t1=sqrt(.5); //0.707

float A,D,E;float3 B,C;

float mav(float2 a){return max(a.y,a.x);}
float mav(float3 a){return max(a.z,mav(a.xy));}
float mav(float4 a){return max(mav(a.zw),mav(a.xy));}
#define miv(a) -mav(-a)
#define dd(a) dot(a,a)
float vsum(float3 a){return dot(a,float3(1,1,1));}//dot() is generally faster on a gpu than 2add()
 //return a.x+a.y+a.z;}

//spaceship distance field is the min() of many sub-distance fields
//sub of H and I
float3 F(float3 a, float b){float c=sin(b),d=cos(b);return mul(float3x3(d,-c,0,c,d,0,0,0,1),a);}
//sub of T,used once
float3 H(float3 a){a=F(a,(floor(atan2(a.x,a.y)*1.5/pi)*2.+1.)*pi/3.);
 return float3(a.x,abs(a.y),a.z);}
//sub of T, used once, modifies a internally though
float3 I(float3 a){a.z-=A*1.5;float b=A*.5 + floor(a.z);
 return F(float3(a.x,a.y+sin(b),frac(a.z)-.5),pi-cos(b));}
//
//sub of S and T
float R(float3 a){float3 b=abs(a);return max(b.y,dot(float3(.87,.5, 0), b))- 1.;}
//sub of T, used twice
float S(float3 a){return max(max(abs(length(a-float3(0,0,5.))-5.)-.05,R(a)),a.z-2.);}
//sub of T, used twice
float Q(float3 a){return max(abs(length(a*float3(1,1,.3))-.325)-.025,-a.z);}
//sub of T,used twice
float P(float3 a){float3 b=abs(a);
 return max(mav(b),max(max(length(b.xy),length(b.yz)),length(b.zx))-.2)-1.;}
//t is most scene specific
//for scene5 it is the distance field of chasing spaceships
float T(float3 a){
 float3 b=I(a)*20.,c=H(b*2.+float3(0,0,2))-float3(1.4,0,0),d=b;
 d.y=abs(d.y);
 return min(min(min(
           min(max(R(d*4.-float3(2,5,0))*.25,abs(d.z)-1.),S(d.yzx*float3(1,.5,.5)*1.5 + float3(.3,0,0))/1.5),
           max(min(.1-abs(d.x),-d.z),S(float3(0, 0, 1) - d.xzy * float3(1, .5, .5)))),
          min(
           min(max(P(c),-P(c * 1.2 + float3(0,0, 1.5)) / 1.2),Q(c + float3(0, 0, 1.5))),
           Q(float3(abs(c.xy), c.z) - float3(.5,.5,-1.5)))*.5)*.05,
  .15-abs(a.x));}

//sub of W and Y
float3 V(float a,float3 b,float c){a*=c;return 1./((1.+2.*b/a+b*b/(a*a))*c+.0001);}
//used twice in Mainimage
float3 W(float3 a,float b,float c,float d){
 float3 e=(V(.01,abs(a),d)*2.+V(.05, float3(length(a.yz),length(a.zx),length(a.xy)),d)*5.)
       *(sin(A * float3(2.1,1.3,1.7)+b*10.0)+1.);
 return(e*7.+e.yzx*1.5+e.zxy*1.5)*max(1.-c*200./d,0.)/d*12.;}

//glowing planes:
//sub of X
float3 Z(float t){
 return float3(0,-sin(t*.6),t*1.6+.5)+sin(t*.01*float3(11,23,19))*float3(.135,.25,.25);}
//sub of Y
float X(float3 a,float t,float b){
 float c=frac(t+b),e=t-c;
 float3 f=Z(e)* float3(0, 1, 1) + sin(float3(0,23,37)*e),
 g=normalize(sin(float3(0, 17, 23) * e))*8.,
 h=f+g+float3(sin(e*53.)*.15,0,9),
 j=f-g+float3(sin(e*73.)*.15,0,9),
 k=lerp(h,j,c-.15),
 l=lerp(h,j,c+.15);
 t=dot(a-k,l-k)/dot(l-k,l-k);
 return length((t<.0?k:t>1.?l:k+t*(l-k))-a);}
//used in main
float4 Y(float3 a,float b,float t){
 float3 c=I(a)*20.,
 d=float3(length(c + float3(-.35,.57,2)),length(c + float3(-.35, -.57, 2)), length(c + float3(.7, 0, 2))),
 e=V(.2,d,b),
 f=float3(X(a, t, 0.0), X(a, t, .3), X(a, t, .6)), g = V(.001, f, b);
 return float4(
  vsum(e)*float3(30, 75, 150) * (E + 1.0) + vsum(g) * float3(1.0, .1, .2) * 5000.0,
  min(min(min(d.y, d.z), d.x) * .05, min(min(f.y, f.z), f.x)));}

//used once in MainImage
float3 G(float3 a, float b){a=frac(a*.2)*2.-1.;a.z=b;float c=50.;
 for(int i=0;i<6+1;++i){//scene5 adds +1 iteration here
  float d = clamp(dd(a),.05,.65);c=mul(c,d);a=abs(a)/d-1.31;a.xy=mul(a.xy,mul(float2x2(1,1,-1,1),t1));
 }return a*c;}
//U is very scene specific, used 5* in mainImage
float U(float3 a){return .15-abs(a.x);}

#define resolution iResolution


         fixed4 frag (v2f v) : SV_Target
            {
                float2 Uuu = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin

 {//this looks like it used to be an initiating function
  A=_Time.y + timeOffset;
  float2 glVertex=Uuu.xy/float2(1,1)*2-1;      //resolution.xy*2.-1.;
  float3 a=Z(A),//a b c d are very scene specific
  b=normalize((float3(0,-sin((A+sin(A*.2)*4.)*.5+A*.1),(A+sin(A*.2)*4.)*1.6+.5)-a)),
  c=normalize(cross(b,sin(A*.001*float3(31,17,29))));
  float d=A*5.;
  for(int i=0;i< 20;++i){
   float t=A-float(i)*.1;
   float4 y=Y(Z(t),25.,t);
   d+=1.*sin((y.w+t)* 5.)*y.x*.05*exp(float(i)*-.25);//scene specific
  }/**/
  float3 e=normalize(float3(sin(float2(.53,.47)*d)*4.+sin(float2(.91,1.1)* d)*2.+sin(float2(2.3,1.7)*d),200)),
  f=normalize(cross(e, float3(sin(d), 50, 0)));
  B=a;
  C=mul(float3x3(c,cross(c,b),b),(f*glVertex.x*1.78+cross(f,e)*glVertex.y+e*1.4));
  D=frac(sin(vsum(C)*99.317*pi)*85.081*pi);
  E=frac(sin(A      *99.317*pi)*85.081*pi);
 }
 float3 a=normalize(C),c=float3(1,1,1),e=B,f=a,g=e,b=float3(0,0,0),s=float3(1,-1,-1)*.0005;
 float4 l=float4(B,1),k=float4(0,0,0,0),j=k,h=j;  
 int m=1;
 float t=.0,o=1.,p=1.,q=D*.01+.99,n;
 for(int i=0;i<64;++i) {//scene5 adds +32 iterations here. 
  //...i removed that. performance loss not wirth that.
  g=e+f*t;
  float d=T(g);
  if(d<(t*5.+1.)*.0001){
   float3 u=normalize(T(g+s)*s+T(g+s.yyx)*s.yyx+T(g+s.yxy)*s.yxy+T(g+s.xxx)*s.xxx);//normal
   float r=pow(abs(1.-abs(dot(u,f))),5.)*.9+.1;
   o+=t*p;p*=5./r;
   e=g+u*.0001;f=reflect(f,u);t=.0;
   float v=dd(u);
   if(v<.9||1.1<v||v!= v)u=float3(0,0,0);
   if(m<4){h=j;j=k;k=l;l=float4(g,max(floor(o),1.)+clamp(r,.001,.999));++m;
   }
  }else t=min(t+d*q,100.);
 }
 if(m<4){h=j;j=k;k=l;l=float4(g,o+t*p);++m;}
 int nn=m;for(int i=0;i<4;++i)if(nn < 4){h=j;j=k;k=l;++nn;}
 f=normalize(j.xyz-h.xyz);n=length(j.xyz-h.xyz);
 t=.0;o=1.;p=.0;e=h.xyz;
 q=D*.1+.8;//scene specific, no mod for scene5
 for(int i=0;i<64+32;++i){//scene5 adds 32 iterations here
  if(t>n){
   if(m<3)break;
   h=j;j=k;k=l;
   --m;
   e=h.xyz;
   f=normalize(j.xyz - h.xyz);
   n=length(j.xyz - h.xyz);
   t=.0;
   if(n<.0001)break;
   float r=frac(h.w);
   o = h.w-r;
   p=(floor(j.w)-o)/n;
   c*=lerp(float3(.17,.15,.12),float3(1,1,1),r);}
  g=e+f*t;
  float4 y=Y(g,o+p*t,A);//scene specific
  float u=U(g);
  u=min(u, y.w);//scene specific
  g-=normalize(U(g+s)*s+U(g+s.yyx)*s.yyx+U(g+s.yxy)*s.yxy+U(g+s.xxx)*s.xxx)*u;
  float v=sin(A*.05+g.z)*.5,w=u*q;//scene specific
  float3 x=G(g,v);//scene specific
  b+=(W(x,v,u,o+p*t)+W(x,v,u,o+p*t+50.)+ y.xyz)*c*w;//scene specific
  c*=pow(.7,w);t+=w;
 }
 //O is scene specific, nno modifier for scene 5.
 fragColor = float4(pow(b, float3(.45,.45,.45)), 1.0);


                return fragColor;
            }

            ENDCG
        }
    }
}

