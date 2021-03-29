
Shader "Skybox/SkyDomeTwinSuns"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// License CC0: Sky dome with twin suns, a gas giant and a rocket launch
//  Through lots of trial and error I created a skydome for another shader.
//  I ended up reasonably satisified with the skydome and extracted it in it's own
//  shader to simplify if someone like to use it or build upon it.
//  Note: There's a distant rocket launch after about 10 sec
    
#define PI  3.141592654
#define TAU (2.0*PI)


const float3  skyCol1       = float3(0.35, 0.45, 0.6);
const float3  skyCol2       = float3(0.0,0,0);
const float3  skyCol3       = pow(float3(0.35, 0.45, 0.6), float3(0.25,.25,.25));
const float3  sunCol1       = float3(1.0,0.6,0.4);
const float3  sunCol2       = float3(1.0,0.9,0.7);
const float3  smallSunCol1  = float3(1.0,0.5,0.25)*0.5;
const float3  smallSunCol2  = float3(1.0,0.5,0.25)*0.5;
const float3  ringColor     = sqrt(float3(0.95, 0.65, 0.45));
const float4  planet        = float4(80.0, -20.0, 100.0, 50.0)*1000.0;
const float3  ringsNormal   = normalize(float3(1.0, 1.25, 0.0));
const float4  rings         = float4(normalize(float3(1.0, 1.25, 0.0)), -dot(normalize(float3(1.0, 1.25, 0.0)), float3(80.0, -20.0, 100.0)*1000));

// From: https://iquilezles.org/www/articles/intersectors/intersectors.htm
float rayPlane(float3 ro, float3 rd, float4 plane) {
  return -(dot(ro,plane.xyz)+plane.w)/dot(rd,plane.xyz);
}

float2 raySphere(float3 ro, float3 rd, float4 sphere) {
  float3 ce = sphere.xyz;
  float ra = sphere.w;
  float3 oc = ro - ce;
  float b = dot(oc, rd);
  float c = dot(oc, oc) - ra*ra;
  float h = b*b - c;
  if (h<0.0) return float2(-1.0,-1); // no intersection
  h = sqrt(h);
  return float2(-b-h, -b+h);
}

float3 sunDirection() {
  return normalize(float3(-0.5, 0.085, 1.0));
}

float3 smallSunDirection() {
  return normalize(float3(-0.2, -0.05, 1.0));
}

float3 rocketDirection() {
  return normalize(float3(0.0, -0.2+(_Time.y% 90.0)*0.0125, 1.0));
}

float psin(float f) {
  return 0.5 + 0.5*sin(f);
}

float3 skyColor(float3 ro, float3 rd) {
  float3 sunDir = sunDirection();
  float3 smallSunDir = smallSunDirection();

  float sunDot = max(dot(rd, sunDir), 0.0);
  float smallSunDot = max(dot(rd, smallSunDir), 0.0);
  
  float angle = atan2( length(rd.xz),rd.y)*2.0/PI;

  float3 skyCol = lerp(lerp(skyCol1, skyCol2, smoothstep(0.0 , 1.0, 5.0*angle)), skyCol3, smoothstep(0.0, 1.0, -5.0*angle));
  
  float3 sunCol = 0.5*sunCol1*pow(sunDot, 20.0) + 8.0*sunCol2*pow(sunDot, 2000.0);
  float3 smallSunCol = 0.5*smallSunCol1*pow(smallSunDot, 200.0) + 8.0*smallSunCol2*pow(smallSunDot, 20000.0);

  float3 dustCol = pow(sunCol2*ringColor, float3(1.75,1.75,1.75))*smoothstep(0.05, -0.1, rd.y)*0.5;

  float2 si = raySphere(ro, rd, planet);
  float pi = rayPlane(ro, rd, rings);
  
  float dustTransparency = smoothstep(-0.075, 0.0, rd.y);
  
  float3 planetSurface = ro + si.x*rd;
  float3 planetNormal = normalize(planetSurface - planet.xyz);
  float planetDiff = max(dot(planetNormal, sunDir), 0.0);
  float planetBorder = max(dot(planetNormal, -rd), 0.0);
  float planetLat = (planetSurface.x+planetSurface.y)*0.0005;
  float3 planetCol = lerp(1.3*float3(0.9, 0.8, 0.7), 0.3*float3(0.9, 0.8, 0.7), pow(psin(planetLat+1.0)*psin(sqrt(2.0)*planetLat+2.0)*psin(sqrt(3.5)*planetLat+3.0), 0.5));

  float3 rocketDir = rocketDirection();
  float rocketDot = max(dot(rd, rocketDir), 0.0);
  float rocketDot2 = max(dot(normalize(rd.xz), normalize(rocketDir.xz)), 0.0);
  float3 rocketCol = float3(0.25,.25,.25)*(3.0*smoothstep(-1.0, 1.0, psin(_Time.y*15.0*TAU))*pow(rocketDot, 70000.0) + smoothstep(-0.25, 0.0, rd.y - rocketDir.y)*step(rd.y, rocketDir.y)*pow(rocketDot2, 1000000.0))*dustTransparency*(1.0 - smoothstep(0.5, 0.6, rd.y));

  float borderTransparency = smoothstep(0.0, 0.1, planetBorder);
  
  float3 ringsSurface = ro + pi*rd;
  float ringsDist = length(ringsSurface - planet.xyz)*1.0;
  float ringsPeriod = ringsDist*0.001;
  const float ringsMax = 150000.0*0.655;
  const float ringsMin = 100000.0*0.666;
  float ringsMul = pow(psin(ringsPeriod+1.0)*psin(sqrt(0.5)*ringsPeriod+2.0)*psin(sqrt(0.45)*ringsPeriod+4.0)*psin(sqrt(0.35)*ringsPeriod+5.0), 0.25);
  float ringslerp = psin(ringsPeriod*10.0)*psin(ringsPeriod*10.0*sqrt(2.0))*(1.0 - smoothstep(50000.0, 200000.0, pi));
  float3 ringsCol = lerp(float3(0.125,.125,.125), 0.75*ringColor, ringslerp)*step(-pi, 0.0)*step(ringsDist, ringsMax)*step(-ringsDist, -ringsMin)*ringsMul;
  
  float3 final = float3(0.0,0,0);
  final += ringsCol*(step(pi, si.x) + step(si.x, 0.0));
  final += step(0.0, si.x)*pow(planetDiff, 0.75)*lerp(planetCol, ringsCol, 0.0)*dustTransparency*borderTransparency + ringsCol*(1.0 - borderTransparency);
  final += skyCol + sunCol + smallSunCol + dustCol + rocketCol;

  return final;
}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

  float2 q = fragCoord.xy; ///iResolution.xy;
  float2 p = -1.0 + 2.0*q;
//  p.x *= iResolution.x/iResolution.y;
  
//  float3 ro  = float3(0.0, 0.0, -2.0);
ro  = ro - float3(0.0, 0.0, -2.0);
  float3 la  = float3(0.0, 0.4, 0.0);

  float3 ww = normalize(la - ro);
  float3 uu = normalize(cross(float3(0.0,1.0,0.0), ww));
  float3 vv = normalize(cross(ww, uu));
//  float3 rd = normalize(p.x*uu + p.y*vv + 2.0*ww);
//rd = normalize(p.x*uu + p.y*vv + 2.0*ww);

  float3 col = skyColor(ro, rd);


  fragColor = float4(col, 1.0);


                return fragColor;
            }

            ENDCG
        }
    }
}



