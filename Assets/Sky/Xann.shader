
Shader "Skybox/Xann"
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

// Created by SHAU - 2017
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
//-----------------------------------------------------

#define T _Time.y * 2.0
#define PI 3.14159265359
#define FAR 1000.0 
#define EPS 0.005
#define TERRAIN 1.0
#define TERRAIN_FLOOR 2.0
#define SPHERE 3.0
#define RODSA 5.0
#define RODSB 6.0
#define CA float3(0.5, 0.5, 0.5)
#define CB float3(0.5, 0.5, 0.5)
#define CC float3(1.0, 1.0, 1.0)
#define CD float3(0.0, 0.33, 0.67)
#define RR 0.8

struct Scene {
    float t;
    float tF;
    float3 n;
    float id;
    float3 sc;
    float rala;
    float rbla;
    float edge;
};
 
float3 cro = float3(0.0,0,0);    
float3 lp = float3(0.0,0,0);

float rand(float2 p) {return frac(sin(dot(p, float2(12.9898,78.233))) * 43758.5453);}
float2x2 rot(float x) {return float2x2(cos(x), sin(x), -sin(x), cos(x));}

float tri(float x) {return abs(x - floor(x) - 0.5);} //Nimitz via Shane
float2 tri(float2 x) {return abs(x - floor(x) - 0.5);}
float layer(float2 p) {return dot(tri(p / 1.5 + tri(p.yx / 3. + .25)), float2(1,1)); }

float noise(float3 rp) {
    float3 ip = floor(rp);
    rp -= ip; 
    float3 s = float3(7, 157, 113);
    float4 h = float4(0.0, s.yz, s.y + s.z) + dot(ip, s);
    rp = rp * rp * (3.0 - 2.0 * rp); 
    h = lerp(frac(sin(h) * 43758.5), frac(sin(h + s.x) * 43758.5), rp.x);
    h.xy = lerp(h.xz, h.yw, rp.y);
    return lerp(h.x, h.y, rp.z); 
}

//IQ cosine palattes
//http://www.iquilezles.org/www/articles/palettes/palettes.htm
float3 palette(float t, float3 a, float3 b, float3 c, float3 d) {
    return a + b * cos(6.28318 * (c * t + d));
}

//IQ distance functions
//http://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
float sdCapsule( float3 p, float3 a, float3 b, float r ){
    float3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

float2 sphIntersect(float3 ro, float3 rd, float4 sph) {
    float3 oc = ro - sph.xyz;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - sph.w * sph.w;
    float h = b * b - c;
    if (h < 0.0) return float2(0.0,0);
    h = sqrt(h);
    float tN = -b - h;
    float tF = -b + h;
    return float2(tN, tF);
}

float planeIntersection(float3 ro, float3 rd, float3 n, float3 o) {
    return dot(o - ro, n) / dot(rd, n);
}

float3 sphNormal(in float3 pos, in float4 sph) {
    return normalize(pos - sph.xyz);
}

//IQ
// cc center, ca orientation axis, cr radius, ch height
float4 iCapsule(float3 ro, float3 rd, float3 cc, float3 ca, float cr, float ch) {
    
    float3  oc = ro - cc;
    ch *= 0.5;

    float card = dot(ca,rd);
    float caoc = dot(ca,oc);
    
    float a = 1.0 - card*card;
    float b = dot( oc, rd) - caoc*card;
    float c = dot( oc, oc) - caoc*caoc - cr*cr;
    float h = b*b - a*c;
    if( h<0.0 ) return float4(-1.0,-1,-1,-1);
    float t = (-b-sqrt(h))/a;

    float y = caoc + t*card;

    // body
    if( abs(y)<ch ) return float4( t, normalize( oc+t*rd - ca*y ) );
    
    // caps
    float sy = sign(y);
    oc = ro - (cc + sy*ca*ch);
    b = dot(rd,oc);
    c = dot(oc,oc) - cr*cr;
    h = b*b - c;
    if( h>0.0 )
    {
        t = -b - sqrt(h);
        return float4( t, normalize(oc+rd*t ) );
    }

    return float4(-1.0,-1,-1,-1);
}

//neat trick from Shane
float2 nearest(float2 a, float2 b){ 
    float s = step(a.x, b.x);
    return s * a + (1. - s) * b;
}

Scene map(float3 rp) {
    
    float3 q = rp;
    q.xz *= 0.8;
    q.y += sin(q.z * 0.05) * 8.; //roll hills
    float ridge = 1.0 / (1.0 + q.x * q.x * 0.002); //central ridge
    q.y -= ridge * 34.0;
    float2 terrain = float2(q.y, TERRAIN);    
    float ax = abs(q.x);
    terrain.x -= ax * ax * 0.003; //valley
    
    //slightly modified logic from Shane (...or breaking it)
    //https://www.shadertoy.com/view/MtdSRn
    float a = 20.0;
    for (int i = 0; i < 3; i++) {
        terrain.x += abs(a) * layer(q.xz / a) * 0.8; 
        q.xz = mul(float2x2(.6, .757, -.577, .98) , q.xz )* 0.6;
        a *= -0.5;     
    }
    float2 near = terrain;

    //rods
    q = rp;
    q.x = abs(q.x);
    float rt = ((T * 80.0)% 1600.0) - 200.0;
    float2 rodsa = float2(FAR, RODSA);
    for (int i = 0; i < 2; i++) {
        float nsc = sdCapsule(q, 
                              float3(50.0 + float(i) * 30.0, 10.0, cro.z + rt + 100.0),
                              float3(50.0 + float(i) * 30.0, 10.0, cro.z + rt - 100.0),
                              RR);
        rodsa.x = min(rodsa.x, nsc);
    }
    
    rt = ((T * 50.0)% 1600.0) - 200.0;
    float2 rodsb = float2(FAR, RODSB);
    for (int i = 0; i < 3; i++) {
        float nsc = sdCapsule(q,
                              float3(35.0 + float(i) * 30.0, 10.0, (cro.z + rt) + (70.0 - float(i) * 20.0)),
                              float3(35.0 + float(i) * 30.0, 10.0, (cro.z + rt) - (70.0 - float(i) * 20.0)),
                              RR);
        rodsb.x = min(rodsb.x, nsc);
        nsc = sdCapsule(q,
                        float3(35.0 + float(i) * 30.0, 10.0, (cro.z + rt - 600.0) + (70.0 - float(i) * 20.0)),
                        float3(35.0 + float(i) * 30.0, 10.0, (cro.z + rt - 600.0) - (70.0 - float(i) * 20.0)),
                        RR);
        rodsb.x = min(rodsb.x, nsc);
    }
    
    Scene scene;
    scene.t = near.x;
    scene.tF = FAR;
    scene.n = float3(0.0,0,0);
    scene.id = near.y;
    scene.sc = float3(-1.0,-1,-1);
    scene.rala = rodsa.x;
    scene.rbla = rodsb.x;
    scene.edge = 0;

    return scene;
}

float3 normal(float3 rp) {
    float2 e = float2(EPS, 0);
    float d1 = map(rp + e.xyy).t, d2 = map(rp - e.xyy).t;
    float d3 = map(rp + e.yxy).t, d4 = map(rp - e.yxy).t;
    float d5 = map(rp + e.yyx).t, d6 = map(rp - e.yyx).t;
    float d = map(rp).t * 2.0;
    return normalize(float3(d1 - d2, d3 - d4, d5 - d6));
}

//Nice bump mapping from Nimitz
float3 bump(float3 rp, float3 n) {
    float2 e = float2(EPS, 0.0);
    float nz = noise(rp);
    float3 d = float3(noise(rp + e.xyy) - nz, noise(rp + e.yxy) - nz, noise(rp + e.yyx) - nz) / e.x;
    n = normalize(n - d * 0.2 / sqrt(0.1));
    return n;
}

//IQ
//http://www.iquilezles.org/www/articles/fog/fog.htm
float3 applyFog(float3  rgb,      // original color of the pixel
              float d, // camera to point distance
              float3  rayDir,   // camera to point floattor
              float3  sunDir,
              float b)  // sun light direction
{
    float fogAmount = 1.0 - exp(-d * b);
    float sunAmount = max(dot(rayDir, sunDir), 0.0);
    float3  fogColor  = lerp(float3(0.5, 0.05, 0.0),
                          float3(0.5, 0.3, 0.8),
                          pow(sunAmount, 16.0));
    return lerp(rgb, fogColor, fogAmount);
}

float3 vMarch1(float3 ro, float3 rd, float3 sc, float3 gc, float maxt) {
 
    float3 pc = float3(0.0,0,0);
    float t = 0.0;
    
    for (int i = 0; i < 128; i++) {
        if (t > maxt) break;
        float3 rp = ro + rd * t;
        float lt = length(sc - rp);
        lt = 0.05 / (1.0 + lt * lt * 0.001);
        //pc += gc * 0.05;
        pc += gc * lt;
        t += 0.5;
    }
    
    return pc;
}

Scene march(float3 ro, float3 rd, float maxt) {
 
    float h = EPS * 2.0;
    float t = 0.0;
    float id = 0.0;
    float rala = 0.0;
    float rbla = 0.0;
    
    for(int i = 0; i < 128; i++) {
        t += h;
        float3 rp = ro + rd * t;
        Scene scene = map(rp);
        if (abs(h) < EPS || t > maxt) {
            id = scene.id;
            break;
        }
        float lat = rp.z - cro.z;
        lat = (1.0 / (1.0 + lat * lat * 0.000001)) * step(3.8, rp.y);;
        rala += 0.3 / (1.0 + scene.rala * scene.rala * 1.0) * lat;
        rbla += 0.3 / (1.0 + scene.rbla * scene.rbla * 1.0) * lat;
        
        h = scene.t * 0.7;
    }
    
    Scene scene;
    scene.t = t;
    scene.tF = FAR;
    scene.n = float3(0.0,0,0);
    scene.id = id;
    scene.sc = float3(-1.0,-1,-1);
    scene.rala = rala;
    scene.rbla = rbla;
    scene.edge = 0;

    return scene;
}

Scene traceScene(float3 ro, float3 rd, float maxt) {
    
    float mint = FAR;
    float mintf = FAR;
    float3 minn = float3(0.0,0,0);
    float id = 0.0;
    float3 sc = float3(-1.0,-1,-1);
    
    //floor
    float3 fo = float3(0.0, 3.8, 0.0);
    float3 fn = float3(0.0, 1.0, 0.0);
    float ft = planeIntersection(ro, rd, fn, fo);
    if (ft > 0.0 && ft < mint) {
        mint = ft;
        minn = fn;
        id = TERRAIN_FLOOR;
    }
    //*/
    
    float rt = ((T * 80.0)% 1600.0) - 200.0;
    for (int i = 0; i < 2; i++) {
        float4 ct = iCapsule( ro, rd, float3(50.0 + float(i) * 30.0, 10.0, cro.z + rt), normalize(float3(0.0, 0.0, 1.0)), RR, 200.0);
        if (ct.x > 0.0 && ct.x < mint) {
            mint = ct.x;
            minn = ct.yzw;
            id = RODSA;
        }
        ct = iCapsule( ro, rd, float3(-50.0 - float(i) * 30.0, 10.0, cro.z + rt), normalize(float3(0.0, 0.0, 1.0)), RR, 200.0);
        if (ct.x > 0.0 && ct.x < mint) {
            mint = ct.x;
            minn = ct.yzw;
            id = RODSA;
        }
    }
    rt = ((T * 50.0)% 1600.0) - 200.0;
    for (int i = 0; i < 3; i++) {
        float4 ct = iCapsule( ro, rd, float3(35.0 + float(i) * 30.0, 10.0, cro.z + rt), normalize(float3(0.0, 0.0, 1.0)), RR, 140.0 - float(i) * 40.0);
        if (ct.x > 0.0 && ct.x < mint) {
            mint = ct.x;
            minn = ct.yzw;
            id = RODSB;
        }
        ct = iCapsule( ro, rd, float3(35.0 + float(i) * 30.0, 10.0, cro.z + rt - 600.0), normalize(float3(0.0, 0.0, 1.0)), RR, 140.0 - float(i) * 40.0);
        if (ct.x > 0.0 && ct.x < mint) {
            mint = ct.x;
            minn = ct.yzw;
            id = RODSB;
        }
        ct = iCapsule( ro, rd, float3(-35.0 - float(i) * 30.0, 10.0, cro.z + rt), normalize(float3(0.0, 0.0, 1.0)), RR, 140.0 - float(i) * 40.0);
        if (ct.x > 0.0 && ct.x < mint) {
            mint = ct.x;
            minn = ct.yzw;
            id = RODSB;
        }
        ct = iCapsule( ro, rd, float3(-35.0 - float(i) * 30.0, 10.0, cro.z + rt - 600.0), normalize(float3(0.0, 0.0, 1.0)), RR, 140.0 - float(i) * 40.0);
        if (ct.x > 0.0 && ct.x < mint) {
            mint = ct.x;
            minn = ct.yzw;
            id = RODSB;
        }
    }
    //*/
    
    //spheres
    //FAR is 800 in front of camera. divide space into 100s
    float dt = (floor((cro.z + FAR) * 0.01) * 100.);
    for (int i = 0; i < 10; i++) {
       
        //large spheres
        float r = rand(float2(20.0, dt - 20.0));
        float4 sphere = float4(20.0 + r * 100.0, 16.0, dt - 20.0, 24.);
        if (r > 0.75) {
            float2 si = sphIntersect(ro, rd, sphere);
            if (si.x > 0.0 && si.x < mint) {
                mint = si.x;
                mintf = si.y;
                minn = sphNormal(ro + rd * si.x, sphere);
                id = SPHERE;
                sc = sphere.xyz;
            }
        }
        
        r = rand(float2(-20.0, dt - 60.0));
        sphere = float4(-20.0 - r * 100.0, 16.0, dt - 60.0, 24.);
        if (r > 0.8) {
            float2 si = sphIntersect(ro, rd, sphere);
            if (si.x > 0.0 && si.x < mint) {
                mint = si.x;
                mintf = si.y;
                minn = sphNormal(ro + rd * si.x, sphere); 
                id = SPHERE;
                sc = sphere.xyz;
            }
        }        
        
        //small spheres
        r = rand(float2(4.0, dt - 5.0));
        sphere = float4(4.0 + r * 60.0, 10.0, dt - 5.0, 12.);
        if (r > 0.75) {
            float2 si = sphIntersect(ro, rd, sphere);
            if (si.x > 0.0 && si.x < mint) {
                mint = si.x;
                mintf = si.y;
                minn = sphNormal(ro + rd * si.x, sphere); 
                id = SPHERE;
                sc = sphere.xyz;
            }
        }        
        
        r = rand(float2(-4.0, dt - 25.0));
        sphere = float4(-4.0 - r * 60.0, 10.0, dt - 25.0, 12.);
        if (r > 0.7) {
            float2 si = sphIntersect(ro, rd, sphere);
            if (si.x > 0.0 && si.x < mint) {
                mint = si.x;
                mintf = si.y;
                minn = sphNormal(ro + rd * si.x, sphere); 
                id = SPHERE;
                sc = sphere.xyz;
            }
        }        
        
        dt -= 100.;    
    }
    //*/
    
    Scene scene;
    scene.t = mint;
    scene.tF = mintf;
    scene.n = minn;
    scene.id = id;
    scene.sc = sc;
    scene.rala = 0;
    scene.rbla = 0;
    scene.edge = 0;

    return scene;
}


float3 colourScene(float3 ro, float3 rd, Scene scene) {
 
    float3 pc = float3(0.0,0,0);
    
    float3 rp = ro + rd * scene.t;
    float3 ld = normalize(float3(160.0, 50.0, 10.));    
    float lt = length(lp - rp);
    float diff = max(dot(ld, scene.n), 0.05);
    float spec = pow(max(dot(reflect(-ld, scene.n), -rd), 0.0), 32.0);
    float atten = 2.0 / (1.0 + lt * lt * 0.0002);
    
    if (scene.id == TERRAIN_FLOOR) {
        
        pc = float3(0.0, 0.04, 0.02) * 0.5 * diff;
        pc += float3(1.0,1,1) * spec * 0.3;
        pc *= atten;
        
    } else if (scene.id == SPHERE) {

        pc = float3(0.1, 0.0, 0.01) * diff;
        pc += float3(1.0,1,1) * spec;
        pc *= atten;
        float3 gc = palette(rand(scene.sc.xz), CA, CB, CC, CD);
        pc += vMarch1(rp, rd, scene.sc, gc, scene.tF - scene.t) * atten;
        
    } else if (scene.id == RODSA) {

        atten = 1.0 / (1.0 + lt * lt * 0.00001);
        pc = float3(1.3, 0.0, 0.0) * atten;
        
    } else if (scene.id == RODSB) {

        atten = 1.0 / (1.0 + lt * lt * 0.00001);
        pc = float3(1.3, 0.8, 0.0) * atten;
        
    } else {
         
        pc = float3(0.01,.01,.01) * diff;
        pc += float3(0.5, 0.0, 0.05) * max(scene.n.z, 0.0) * 0.15;
        pc += float3(0.05, 0.0, 0.3) * max(scene.n.z * -1.0, 0.0) * 0.02;
        pc += float3(0.05, 0.0, 0.3) * max(scene.n.x, 0.0) * 0.02;
        pc += float3(1.0,1,1) * spec * 0.6;
        pc *= atten;
    
    }
    
    return pc;
}

void setupCamera(float2 uv, inout float3 rd) {

    float3 lookAt = float3(0.0, 28.0, T * 40.);
    cro = lookAt + float3(0.0, 30.0, -54.0);
    lp = lookAt + float3(50.0, 50.0, 10.);
    
    float FOV = PI / 3.0;
    float3 forward = normalize(lookAt - cro);
    float3 right = normalize(float3(forward.z, 0.0, -forward.x)); 
    float3 up = cross(forward, right);

    rd = normalize(forward + FOV * uv.x * right + FOV * uv.y * up);
}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin

    float3 pc = float3(0.0,0,0);
    float mint = FAR;
//    float2 uv = (fragCoord.xy - iResolution.xy * 0.5) / iResolution.y;
    float3 rala = float3(0.0,0,0);
    float3 rbla = float3(0.0,0,0);
    bool glow = true;
    float terrainDepth;
    
  //  float3 rd;
  //  setupCamera(uv, rd);
    
    cro = ro;
    Scene terrain = march(cro, rd, FAR);
    if (terrain.t > 0.0 && terrain.t < mint) {
        mint = terrain.t;
        float3 rp = cro + rd * terrain.t;
        terrainDepth = rp.y;
        terrain.n = normal(cro + rd * terrain.t);
        pc = colourScene(cro, rd, terrain);        
    }

    rala = float3(1.0, 0.0, 0.0) * terrain.rala;
    rbla = float3(1.0, 0.5, 0.0) * terrain.rbla;
    //*/
    
    Scene scene = traceScene(cro, rd, FAR);
    if (scene.t < mint) {
        
        mint = scene.t;
        float3 rp = cro + rd * (scene.t - EPS);
        float3 n = scene.n;
        
        if (scene.id == TERRAIN_FLOOR) {
            pc = lerp(colourScene(cro, rd, scene), pc, 0.2 / (1.0 + rp.y - terrainDepth)); 
            n = bump(rp * 0.6 + T, n);
        } else {
            pc = colourScene(cro, rd, scene);
        }
        
        if (scene.id == SPHERE) glow = false;
        
        //TODO: Read up on how to do this properly
        float3 rrd = reflect(rd, n);
        Scene reflectedScene = traceScene(rp, rrd, 200.0);
        if (reflectedScene.t < FAR) {
            float tt = scene.t + reflectedScene.t;
            float atten = 0.5 / (1.0 + tt * tt * 0.0005);
            float3 rc = colourScene(rp, rrd, reflectedScene);
            if (scene.id == SPHERE && (reflectedScene.id == RODSA || reflectedScene.id == RODSB)) {
                rc /= scene.t * 0.01;
                pc = lerp(pc, pc + rc, 1.0 - (reflectedScene.t / FAR));
            } else {
                rc *= atten;
                pc += rc;
            }
        }
    }
    
    if (glow) {
        pc += rala;
        pc += rbla;
    }
    
    pc = applyFog(pc, mint, rd, normalize(float3(4.0, 5.0, 2.0)), 0.0005);  
    
    fragColor = float4(sqrt(clamp(pc, 0.0, 1.0)), 1.0);


                return fragColor;
            }

            ENDCG
        }
    }
}


