
Shader "Skybox/ShatteredDojo22"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _SunDir ("Sun Dir", Vector) = (-.11,.07,0.99,0) 
        _XYZPos ("XYZ Offset", Vector) = (10.,10.,-20. ,0) 
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



// Shattered dimention
// Inspired by the Riftborn intro from ES2
// https://www.youtube.com/watch?v=h8H8DedCW_I
//
// Disclamier: I work at Amplitude at the moment =)

#define time (_Time.y)

#define PI 3.14159
#define TAU (PI * 2.)

#define RID(p, r) (floor((p + r/2.) / r))
#define REP(p, r) ((p + r/2.)%r - r/2.)      //(mod(p + r/2., r) - r/2.)

float hash( float n ) {
    return frac(sin(n)*43758.5453);
}

float noise( in float3 x ) { // in [0,1]

    float res = (hash(x.x) + hash(x.y + 1523.) + hash(x.z + 423.)) / 3.0;
    return res;
}

float2x2 rot(float a)
{
  float ca = cos(a); float sa = sin(a);
  return float2x2(ca,-sa,sa,ca);
}

float sdPlane(float3 p, float4 n)
{
  return dot(p,n.xyz) - n.w;
}

float solid(float3 p, float s)
{
  float dist = -1000.;
  float3 dir = float3(0.,1.,0.);
  float h = 3.;
  float v = 6.;
  float2x2 hr = rot(TAU / h);
  float2x2 vr = rot(TAU / v);

  for(float j = 0.; j < v; ++j)
  {
    for(float i = 0.; i < h; ++i)
    {
      float ran = (hash(s + j * i)-.5) ;
      dist = max(dist,sdPlane(p, float4(dir,1. + ran)));
      dir.xz = mul(dir.xz,hr);
    }

    dir.yz = mul(dir.yz,vr);
  }

  return dist;
}

float map(float3 p)
{

  float3 cp = p;

  float3 pid = RID(p, 80.);
  p = REP(p, 80.);
  p.xy = mul(p.xy,rot(pid.z * .5));
  p.zy = mul(p.zy,rot(pid.x * .5));
  p.xz = mul(p.xz,rot(pid.y * .5));

  p.z -= time;

  p.y += 2.;
  p.xy = mul(p.xy,rot(p.z * .1));
  p.y -= 2.;

  p.xy = mul(p.xy,rot(time * .05));

  float dist = 1000.;

  p *= 2.;
  float r = 6.;

  float3 id = RID(p, r);
  p = REP(p , r);


  float nois = noise(id);
  float t = time * (nois + .2); ;

  p.xz = mul(p.xz,rot(t));
  p.xy = mul(p.xy,rot(t * .5));
  p.yz = mul(p.yz,rot(t * .25));
  dist = solid(p, nois) *.25;

  dist = max(dist, length(id.xy) - 2.);

  return dist;
}

void ray(inout float3 cp,float3 rd, out float st, out float cd)
{
  for(st = 0.; st < 1.; st += 1./200.)
  {
    cd = map(cp);
    if(cd < .01 || cd > 60.)
      break;
    cp += rd * cd * st;
  }
}  

float3 normal(float3 p)
{
  float2 e = float2(.05,.0);
  float d = map(p);
  return normalize(float3(
    d - map(p + e.xyy),
    d - map(p + e.yxy),
    d - map(p + e.yyx)
  ));
}

float3 lookAt(float3 eye, float3 tar, float2 uv)
{
  float3 fd = normalize(tar - eye);
  float3 ri = cross(fd, float3(0.,1.,0.));
  float3 up = cross(ri,fd);
  return normalize(fd + ri * uv.x + up * uv.y);
}

         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+_XYZPos;                                             // ray origin

 // float2 uv = float2(fragCoord.x / iResolution.x, fragCoord.y / iResolution.y);
 // uv -= 0.5;
//  uv /= float2(iResolution.y / iResolution.x, 1);

  float3 eye = ro; //float3(10.,10.,-20.);
  float3 tar = float3(0.,0,0);
  //rd = lookAt(eye, tar, uv);
  float3 cp = eye;
  float st,cd;
  ray(cp,rd,st,cd);

  float dist = length(eye - cp);

  fragColor = lerp(float4(1.,1,1,1), float4(.6,.6,.65,1.), pow((fragCoord.x / 800 * .75 + .1), 1.2));
  if(cd < .01)
  {
    float3 norm = normal(cp);
    float3 ld = normalize(float3(1.,-1.,1.));
    float li = dot(norm,ld);
    fragColor = lerp(
        fragColor,
        lerp(float4(.75,.75,.74999,1.), float4(.4,.4,.52,1.), li),
        exp(-distance(cp,eye) * .007));
  }
  
  fragColor = pow(fragColor, float4(2.2,2.2,2.2,2.2));


                return fragColor;
            }

            ENDCG
        }
    }
}


