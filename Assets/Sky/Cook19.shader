
Shader "Skybox/Cookie19"
{
    Properties
    {
        _MainTex ("tex2D", 2D) = "white" {}
        _SunDir ("Cloud Color", Vector) = (-.11,.07,0.99,0) 
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
                float4 vertex : SV_POSITION;   //pos
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                return o;
            }

// Winning shader made at COOKIE Party 2019 Shader Showdown,
// 1st round against LJ

// The "Shader Showdown" is a demoscene live-coding shader battle competition.
// 2 coders battle for 25 minutes making a shader on stage. No google, no cheat sheets.
// The audience votes for the winner by making noise or by voting on their phone.

// "I'm a marvelous housekeeper: every time I leave a man, I keep his house." Zaza Gabor

float2 z,e=float2(.0035,-.0035);float tt,g,bb,noi,mm;float3 al,po,no,ld;float4 np;
float bo(float3 p,float3 r){p=abs(p)-r;return max(max(p.x,p.y),p.z);}
float2x2 r2(float r){return float2x2(cos(r),sin(r),-sin(r),cos(r));}
float smin(float a,float b,float h){float k=clamp((a-b)/h*.5+.5,0.,1.);return lerp(a,b,k)-k*(1.-k)*h;}
float2 smin2(float2 a,float2 b,float h){float k=clamp((a.x-b.x)/h*.5+.5,0.,1.);return lerp(a,b,k)-k*(1.-k)*h;}
// rough shadertoy approximation of the bonzomatic noise tex2D by yx https://www.shadertoy.com/view/tdlXW4
float4 texNoise(float2 uv){ float f = 0.; f+=tex2D(_MainTex, uv*.125).r*.5;
    f+=tex2D(_MainTex,uv*.25).r*.25;f+=tex2D(_MainTex,uv*.5).r*.125;
    f+=tex2D(_MainTex,uv*1.).r*.125;f=pow(f,1.2);return float4(f*.45+.05,f*.45+.05,f*.45+.05,f*.45+.05);
}
float2 synapse(float3 p){
  p.x-=sin(p.y*15.+tt*10.)*0.03;
  noi=texNoise(float2(.1,.2)*float2(p.y,dot(p.xz,float2(.7,.7)))).r;
  bb=cos(p.y*.25+tt*2.)*.5+.5;
  float  h,t=length(p.xz-(sin(p.xz*1.5-tt)*.5+cos(p.y*2.+noi*5.-tt)*.3))-(1.+bb*2.);
  p=abs(p)-(1.+bb*2.);
  p.xy=mul(p.xy,r2(-.5));
  p.yz=mul(p.yz,r2(.5));
  p+=cos(p.y)*.2;
  h=length(p.xz)-(.4+.3*sin(p.y*.5+noi*3.)+.3*(.5+.5*sin(p.y*2.-tt*3.)));
 t=smin(t,h,.5);
  return float2(t*.5,bb+.15);
}
float2 mp( float3 p )
{
  np=float4(p,1);
  float2 h,t=synapse(p);
  mm=sin(tt-p.y*.5)*2.;
  for(int i=0;i<2;i++){
    np*=2.;
    np.xyz=abs(np.xyz)-float3(8.5+mm,8.5,8.5+mm);
    np.xy=mul(np.xy,r2(-.4));
    np.yz=mul(np.yz,r2(.2));
    h=synapse(np.xyz);
    h.x/=np.w;
    t=smin2(t,h,0.5);
  }
  np.xz=mul(np.xz,r2(.785));
  h=float2(bo(abs(np.xyz)-3.*mm,float3(0,100,0)),1.);
  g+=.1/(.1+h.x*h.x*.2);
  h.x=.7*h.x/np.w;
  t=t.x<h.x?t:h;
  bb=texNoise(p.xz*.05).r*.5;
  h=float2(0.45*length(p+float3(0,81.5,0)-mm+bb*3.)-32.,1.-bb*3.);
  t=smin2(t,h,.5);
  h=float2(length(cos(np.xyz*.2-tt))-.0,1);
  g+=.1/(.1+h.x*h.x*50.);
  t=t.x<h.x?t:h;
  return t;
}
float2 tr( float3 ro,float3 rd )
{
  float2 h,t=float2(.1,.1);
  [loop]
  for(int i=0;i<64;i++){             //for(int i=0;i<128;i++){
    h=mp(ro+rd*t.x);
    if(h.x<.0001||t.x>30.) break;
    t.x+=h.x;t.y=h.y;
  }
  if(t.x>30.) t.y=0.;
  return t;
}
#define a(d) clamp(mp(po+no*d).x/d,0.,1.)
#define s(d) smoothstep(0.,1.,mp(po+ld*d).x/d)

         fixed4 frag (v2f v2) : SV_Target
            {
                float2 fragCoord = v2.vertex;

                float3 viewDirection = normalize(v2.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v2.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+_XYZPos;                                             // ray origin

  //float2 uv=(fragCoord.xy/iResolution.xy-0.5)/float2(iResolution.y/iResolution.x,1);
  tt=(_Time.y+8.)%62.83;
  //float3 ro=float3(cos(tt*.2)*14.,-5.+cos(tt*.2)*5.,sin(tt*0.2)*14.),
  //cw=normalize(float3(0,sin(tt*.4)*10.,0)-ro),
  //cu=normalize(cross(cw,float3(0,1,0))),
  //cv=normalize(cross(cu,cw)),
  //rd=float3x3(cu,cv,cw)*normalize(float3(uv,.5))
  float3 co,fo;
  ld=normalize(float3(.1,.3,0));
  float2 v=float2(abs(atan2(rd.x,rd.z)),rd.y);
  co=fo=float3(.17,.2,.2)+texNoise(v*.4).r*.5;
  z=tr(ro,rd);
  if(z.y>0.){
    po=ro+rd*z.x;
    no=normalize(e.xyy*mp(po+e.xyy).x+
    e.yyx*mp(po+e.yyx).x+
    e.yxy*mp(po+e.yxy).x+
    e.xxx*mp(po+e.xxx).x),al=float3(.4+bb*.5,.4,.4-bb*.1);
    al=lerp(float3(.3,.6,.9),float3(.6,.3,.2),z.y);
    float di=max(0.,dot(no,ld)),
    fr=pow(1.+dot(no,rd),4.),
    sp=pow(max(dot(reflect(-ld,no),-rd),0.),30.);
    co=lerp(sp+al*(a(.2)*a(.4)+.2)*(di+s(.4)+s(2.)),fo,min(fr,.5));
    co=lerp(fo,co,exp(-.0002*z.x*z.x*z.x));
  }
  fragColor = float4(pow(co+g*.1,float3(.45,.45,.45)),1);

                return fragColor;
            }

            ENDCG
        }
    }
}

