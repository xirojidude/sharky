
Shader "Skybox/CityFlight"
{
    Properties
    {
        _MainTex1 ("Texture", 2D) = "white" {}
        _MainTex2 ("Texture", 2D) = "white" {}
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

            uniform sampler2D _MainTex1,_MainTex2; 
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


// -----------
// CITY FLIGHT
// -----------
// by Alexis THIBAULT
//
// Flying through a city using simple raymarching


// This shader was inspired by the Disney Animation paper:
//
// Norman Moses Joseph, Brett Achorn, Sean D. Jenkins, and Hank Driskill.
// Visualizing Building Interiors using Virtual Windows.
// In ACM SIGGRAPH Asia 2014 Technical Briefs, December 2014.


// Stuff that uses similar ideas
// (marching a regular grid by raytracing the boxes, and
//  putting stuff inside)
//
// "Cubescape" by iq
// https://www.shadertoy.com/view/Msl3Rr


// Update - adjusted the max number of iterations: it runs even faster now!
//        - Use the "volume" noise texture instead of the 2D one in hash()
//            (which is much more logical).
//        - Added stars in the sky!


// Should the windows be flashing?
#define PARTY_MODE 0


float2 boxIntersect(float3 ro, float3 rd, float3 r, out float3 normal, out float3 normal2)
{
    // Compute intersection time t of ray ro+t*rd
    // with a box of half-size r centered at 0.
    ro *= sign(rd);
    float3 t1 = (-r-ro)/abs(rd);
    float3 t2 = (r-ro)/abs(rd);
    float tmin = max(t1.x, max(t1.y, t1.z));
    float tmax = min(t2.x, min(t2.y, t2.z));
    normal = -sign(rd) * step(t1.yzx,t1.xyz) * step(t1.zxy, t1.xyz);
    normal2 = -sign(rd) * step(t2.xyz,t2.yzx) * step(t2.xyz, t2.zxy);
    if(tmax < tmin) return float2(-1.,-1);
    return float2(tmin, tmax);
}

float3 hash(float3 p)
{
    return  tex2Dlod(_MainTex1, float4((floor(p)+0.5)*(1./32.), 0.)).xyz; //textureLod(_MainTex1, (floor(p)+0.5)*(1./32.), 0.).xyz;
}

#if 0
float3 sun = normalize(float3(1.0, 1., 0.2));
float3 sunCol = float3(1.,0.5,0.2);
float3 skyCol = float3(0.4,0.65,1.0);
float3 horizonCol = float3(0.6,0.7,0.8);
#else
float3 sun = normalize(float3(1.0, 1., 0.2));
float3 skyCol = float3(0.01,0.02,0.07);
float3 horizonCol = float3(0.002,0.005,0.02);
#endif

const float T_MAX = 1000.;
const float FLOOR = -80.;
const float HEIGHT = 20.;
const float3 blockSize = float3(5.,5.,1000.);

float3 skyColor(float3 rd)
{
    #if 0
    float3 col = skyCol;
    float3 horiz = lerp(horizonCol, 2.5*sunCol, smoothstep(-1.,1.,dot(rd,sun)));
    float horizHeight = 0.1*exp(-2.*(1.3-dot(sun, rd)));
    col = lerp(col, horiz, exp(-max(rd.z,0.)/horizHeight));
    col *= exp(min(rd.z,0.)*15.);
    #else
    float3 col = skyCol;
    float horizHeight = 0.1;
    col = lerp(col, horizonCol, exp(-max(rd.z,0.)/horizHeight));
    col *= exp(min(rd.z,0.)*15.);
    #endif
    return col;
}

void getCurrentBuilding(float3 ro, out float3 boxC,
                        out float3 buildingC, out float3 buildingSize)
{
    boxC = 2.*blockSize*round(ro/(2.*blockSize));
    buildingC = float3(boxC.xy, -2.*HEIGHT) + float3(2.,2.,0.)*(2.*hash(boxC.zxy)-1.);
    float2 maxSize = 4.5-abs(buildingC.xy-boxC.xy);
    buildingSize = float3(1,1,2.*HEIGHT) + float3(maxSize.xy-1.,13.)*hash(boxC.yzx);
}


float sceneIntersect(float3 ro, float3 rd, out float3 normal)
{
    float t = 0.;
    float t0 = 0.;
    float3 boxC = float3(0,0,0);
    int i;
    float3 p;
    const int ITER = 40;
    float3 buildingC, buildingSize;
    float3 _; // Dummy variable
    for(i=0; i<ITER; i++)
    {
        // Intersect building in current box
        getCurrentBuilding(ro, boxC, buildingC, buildingSize);
        t = boxIntersect(ro-buildingC,
                         rd, buildingSize, normal, _).x;
        
        // Intersect current box itself
        float t1 = boxIntersect(ro-boxC, rd, blockSize, _, _).y;
        
        // Also intersect a floor plane and a sky plane
        float tfloor = -(ro.z-FLOOR)/rd.z;
        if(tfloor < 0.) tfloor = 1e5;
        float tsky = -(ro.z - 20.)/rd.z;
        if(tsky < 0.) tsky = 1e5;
        
        if(t > 0. && t < t1 && t < tfloor)
        {
            // We hit the building!
            //p = ro+t*rd;
            return t0+t;
            break;
        }
        else if(tfloor > 0. && tfloor < t1)
        {
            // We hit the floor!
            //p = ro+tfloor*rd;
            return T_MAX;
            normal = float3(0,0,1);
            return t0+tfloor;
        }
        else if(tsky > 0. && tsky < t1)
        {
            // We hit the ceiling!
            return T_MAX;
        }
        // We hit nothing : march to the next cell
        ro += (t1+0.001)*rd;
        t0 += t1;
        p = ro+t1*rd;
        continue;
    }
    return T_MAX;
}

void getRoom(float3 p, float3 rd, out float3 roomSize, out float3 roomCenter,
            out float3 roomHash)
{
    
    float3 boxC, buildingC, buildingSize;
    getCurrentBuilding(p, boxC, buildingC, buildingSize);

    roomSize = buildingSize/(2.*round(buildingSize)+1.);
    roomCenter = round((p-buildingC)/(2.*roomSize) + 0.1*rd)*2.*roomSize + buildingC;
    roomHash = hash(roomCenter*10.);
}

float3 someNoise(float3 p)
{
    p *= 10.;
    float3 v = hash(p)*0.5;
    p.xyz = p.yzx*1.62;
    v += hash(p)*0.25;
    p.xyz = p.yzx*1.62;
    v += hash(p)*0.125;
    return v;
}


float3 computeEmission(float3 p, float3 rd, float t, float3 normal,
                    out float isInWindow)
{

    // Window emission depends on the ray direction...
    // because we actually look at what's inside the room!
    float3 roomSize, roomCenter, roomHash;
    getRoom(p, rd, roomSize, roomCenter, roomHash);
    float3 roomHash2 = hash(roomCenter.yzx);
    float3 q = abs(p-roomCenter);
    float3 inNormal, _;
    float2 inT = boxIntersect(p-roomCenter, rd, roomSize, _, inNormal);
    float3 roomP = p+inT.y*rd;
    
    float border = 0.1;
    float muntins = roomHash2.z > 0.5 ? 0.01 : -0.1;
    float w = t/(450*dot(normal,-rd)); //t/(iResolution.y*dot(normal,-rd)); // A little anti-aliasing
    isInWindow = (smoothstep(q.x-w, q.x+w, roomSize.x-border) * smoothstep(q.x+w, q.x-w, muntins)
                + smoothstep(q.y-w, q.y+w, roomSize.y-border) * smoothstep(q.y+w, q.y-w, muntins))
                * smoothstep(q.z-w, q.z+w, roomSize.z-border) * smoothstep(q.z+w, q.z-w, muntins)
                * step(-0.5,-normal.z);

    #if PARTY_MODE
    float thresh = 0.8 + 0.1*sin(6.28*iTime+6.28*roomHash.r);
    float thresh2 = 0.85 + 0.05*sin(6.28*iTime+6.28*roomHash.g);
    #else
    float thresh=0.8, thresh2=0.85;
    #endif
    float3 emission = float3(1.,0.7,0.5)*smoothstep(thresh-0.1,thresh+0.2,roomHash.g)
        + float3(0.5,0.8,1.)*0.8*smoothstep(thresh2-0.1,thresh2+0.1,roomHash.b);
    emission *= emission*3.0;

    //emission = 0.5+0.5*inNormal;
    //emission = (roomCenter - (ro+inT.y*rd));
    float3 noise = someNoise(roomP);
    float3 randomColor = lerp(roomHash2, 1.-roomHash2.yzx, smoothstep(roomHash.x, roomHash.y,noise.rrr));
    float3 wallColor = dot(inNormal, 2.*roomHash-1.) > 0. ? float3(1.0,1,1) : randomColor;
    if(inNormal.z > 0.9)
    {
        // Floor is same color as the light, with some pattern
        wallColor = emission*0.3;
        float3 floorP = roomHash2.y > 0.5 
            ? float3(roomP.x+roomP.y,roomP.x-roomP.y,roomP.z)*0.7
            : roomP;
        wallColor *= someNoise(floorP).rrr;
    }
    wallColor += 0.5;
    
    // Make ceiling not too bright
    wallColor = lerp(wallColor*2., float3(0,0,0), 0.9*smoothstep(-roomSize.z, roomSize.z, roomP.z-roomCenter.z));
    float3 ligPos = roomCenter + 0.7*float3(roomHash2.xy*2.-1.,1.)*roomSize;
    float intensity = 0.5/dot(ligPos-roomP,ligPos-roomP);
    float3 insideLight = emission * clamp(dot(inNormal, normalize(ligPos-roomP)),0.,1.) * intensity * wallColor;
    insideLight += 0.5*roomHash2*roomHash2*emission;
    
    // Some windows have "curtains"... but you can peek through :)
    float2 curtainW = roomSize.xy - 0.15;
    float curtains = roomHash.x > 0.8 
        ? 0.9*(smoothstep(q.x-w, q.x+w, curtainW.x) + smoothstep(q.y-w, q.y+w, curtainW.y))
        : 0.0;
    emission = lerp(emission*(1.+roomHash2)*0.1, insideLight, 1.-curtains);
    
    emission *= isInWindow;
    return emission;
}

float3 raycast(float3 ro, float3 rd)
{
    float3 normal, normal2;
    float t = sceneIntersect(ro, rd, normal);
    float3 p = ro+t*rd;
    // And after one bounce
    float3 ro2 = p + 0.01*normal, rd2 = reflect(rd, normal);
    float t2 = sceneIntersect(ro2, rd2, normal2);
    
    //return (t < T_MAX) ? 0.5+0.5*normal : float3(0);
    
    float3 _; // Dummy variable
    if(t < T_MAX)
    {
        // Let's do some shading!
        float ao = 0.5+0.5*normal.z;
        float darkVoid = smoothstep(FLOOR,0.,p.z);
        ao *= darkVoid;
        
        float isInWindow, _;
        float3 emission = computeEmission(p, rd, t, normal, isInWindow);
        float3 emission2 = computeEmission(ro2+t2*rd2, rd2, t+t2, normal2, _);
        emission2 = t2 < T_MAX ? emission2 : skyColor(rd2);
        
        float3 roomSize, roomCenter, roomHash;
        getRoom(p, rd, roomSize, roomCenter, roomHash);
        float3 surfCol = 0.4+0.5*smoothstep(0.5,0.9,roomHash.rrr)-0.2*isInWindow;
        float3 F0 = 0.04+float3(0.04,0.1,0.2)*surfCol;
        
        float3 fre = F0 + (1.-F0) * pow(clamp(1.-dot(-rd,normal),0.,1.),5.);
        float3 col = lerp(emission, emission2, fre);
        col = lerp(float3(0.,0,0), col, isInWindow);
        //float3 col = emission;
        //col += surfCol* clamp(dot(normal, sun), 0., 1.) * sunCol  * darkVoid;
        //col += surfCol* clamp(dot(normal, -rd), 0., 1.)  * darkVoid * 2./ t;
        //col += surfCol*float3(0.5,0.7,0.9) * 0.1 * (0.5+0.5*normal.z) * ao;
        //col += surfCol* sunCol*sunCol * 0.1 * clamp(dot(normal.xy, -sun.xy), 0., 1.) * ao;
        
        col += surfCol * ao * 0.2 * (0.5-0.2*normal.z) 
            * smoothstep(10.0,-30.0,p.z) * float3(1.,0.8,0.6);
        
        col = lerp(col, float3(0.,0,0), 1.-exp(-t*0.003));
        col = lerp(col, skyColor(rd), 1.-exp(-t*0.01));
        col *= exp(0.02*min(p.z,0.));
        //col +=;
        return col;
    }
    else
    {
        float3 col = skyColor(rd);
        // Stars
        //smoothstep(0.96,1.01,textureLod(_MainTex2,(round(1.5*rd.yz/(rd.x+1.)*iResolution.y)+0.5)/256., 0.).r)
        float stars = smoothstep(0.96,1.01,      tex2Dlod(_MainTex2,     float4((round(1.5*rd.yz/(rd.x+1.)*450)+0.5)/256.,0, 0.) ).r         )
            * smoothstep(0.0,1.0,tex2Dlod(_MainTex2, float4((rd.yz/(rd.x+1.)*450.)/256., 0.,0)   ).r);
        col += stars*stars*smoothstep(0.0,0.2,rd.z);
        return col;
    }
}

float3 path(float t)
{
    //return float3(20.*t, 20.0*sin(0.3*t), 5.0 - 2.*cos(0.5*t));
    return float3(20.*t, 5.+0.1*cos(0.5*t),10.*cos(0.5*t)*(1.-0.5*sin(0.1*t)));
}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex1, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+_XYZPos.xyz;                                             // ray origin

//   float2 uv = (2.*fragCoord-iResolution.xy)/iResolution.y;
    
    //float3 ro = float3(-3.*cos(th),-3.*sin(th),2.0);
    //float3 target = float3(0);
 //   float3 ro = path(iTime);
//    float3 target = path(iTime+3.)+float3(0.,20.*cos(0.3*iTime),-15.);
//    float3 camFwd = normalize(target - ro);
//    float3 camRight = normalize(cross(camFwd, float3(0.3*cos(0.2*iTime),0,1))); // Camera tilts
//    float3 camUp = cross(camRight, camFwd);
//    float3 rd = normalize(camFwd + 0.7*(uv.x*camRight+uv.y*camUp));
    
    
    float3 col = raycast(ro, rd);
    
    // Vignette
//    col *= smoothstep(1.7,0.5,length(2.*fragCoord/iResolution.xy-1.));
    // Tone mapping
//    col = lerp(col, 1.-(4./27.)/(col*col), step(2./3.,col));
    // Gamma correction
//    col = pow(col, float3(0.45,0.45,0.45));
    
    fragColor = float4(col,1.0);


                return fragColor;
            }

            ENDCG
        }
    }
}
